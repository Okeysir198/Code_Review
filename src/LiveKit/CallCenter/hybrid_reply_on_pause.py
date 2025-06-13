"""
Enhanced ReplyOnPause with LiveKit Audio Processing
Integrates LiveKit's noise cancellation and turn detection into FastRTC
"""

import asyncio
import inspect
from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass, field
from logging import getLogger
from threading import Event
from typing import Any, Literal, cast, Optional

import numpy as np
from numpy.typing import NDArray
import noisereduce as nr

# FastRTC imports (existing)
from .pause_detection import ModelOptions, PauseDetectionModel, get_silero_model
from .tracks import EmitType, StreamHandler
from .utils import AdditionalOutputs, WebRTCData, create_message, split_output

# LiveKit imports for audio processing
try:
    from livekit.plugins import silero as livekit_silero
    from livekit.rtc import AudioFrame
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    AudioFrame = None

logger = getLogger(__name__)


@dataclass
class LiveKitAudioConfig:
    """Configuration for LiveKit audio processing features."""
    use_livekit_vad: bool = True
    use_noise_cancellation: bool = True
    noise_reduction_strength: float = 0.8
    use_echo_cancellation: bool = True
    vad_min_speech_duration: float = 0.1
    vad_min_silence_duration: float = 0.3
    vad_padding_duration: float = 0.1
    vad_activation_threshold: float = 0.6


@dataclass
class AlgoOptions:
    """
    Algorithm options.

    Attributes:
    - audio_chunk_duration: Duration in seconds of audio chunks passed to the VAD model.
    - started_talking_threshold: If the chunk has more than started_talking_threshold seconds of speech, the user started talking.
    - speech_threshold: If, after the user started speaking, there is a chunk with less than speech_threshold seconds of speech, the user stopped speaking.
    - max_continuous_speech_s: Max duration of speech chunks before the handler is triggered, even if a pause is not detected by the VAD model.
    - livekit_config: Optional LiveKit audio processing configuration.
    """

    audio_chunk_duration: float = 0.6
    started_talking_threshold: float = 0.2
    speech_threshold: float = 0.1
    max_continuous_speech_s: float = float("inf")
    livekit_config: Optional[LiveKitAudioConfig] = None


@dataclass
class AppState:
    stream: np.ndarray | None = None
    sampling_rate: int = 0
    pause_detected: bool = False
    started_talking: bool = False
    responding: bool = False
    stopped: bool = False
    buffer: np.ndarray | None = None
    responded_audio: bool = False
    interrupted: asyncio.Event = field(default_factory=asyncio.Event)

    def new(self):
        return AppState()


class LiveKitAudioProcessor:
    """
    LiveKit audio processing integration for noise cancellation and enhanced VAD.
    """
    
    def __init__(self, config: LiveKitAudioConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        
        # Initialize LiveKit VAD if available and enabled
        self.livekit_vad = None
        if LIVEKIT_AVAILABLE and config.use_livekit_vad:
            try:
                self.livekit_vad = livekit_silero.VAD.load(
                    min_speech_duration=config.vad_min_speech_duration,
                    min_silence_duration=config.vad_min_silence_duration,
                    padding_duration=config.vad_padding_duration,
                    sample_rate=sample_rate,
                    activation_threshold=config.vad_activation_threshold
                )
                logger.info("LiveKit VAD initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize LiveKit VAD: {e}")
                self.livekit_vad = None
        
        # Audio processing state
        self.last_speech_probability = 0.0
        self.noise_profile = None
        self.noise_sample_duration = 1.0  # seconds of initial audio to use for noise profile
        self.noise_samples_collected = 0
        self.noise_buffer = []
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio with noise cancellation and echo suppression.
        """
        processed_audio = audio_data.copy()
        
        # Apply noise cancellation if enabled
        if self.config.use_noise_cancellation:
            processed_audio = self._apply_noise_cancellation(processed_audio)
        
        # Apply echo cancellation if enabled (simplified)
        if self.config.use_echo_cancellation:
            processed_audio = self._apply_echo_suppression(processed_audio)
        
        return processed_audio
    
    def _apply_noise_cancellation(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply LiveKit-style noise cancellation using spectral gating.
        """
        try:
            # Collect noise profile from initial silence
            if self.noise_profile is None and self.noise_samples_collected < self.sample_rate * self.noise_sample_duration:
                self.noise_buffer.extend(audio_data.tolist())
                self.noise_samples_collected += len(audio_data)
                
                # Once we have enough samples, create noise profile
                if self.noise_samples_collected >= self.sample_rate * self.noise_sample_duration:
                    noise_sample = np.array(self.noise_buffer[:int(self.sample_rate * self.noise_sample_duration)])
                    self.noise_profile = noise_sample
                    self.noise_buffer = []  # Clear buffer
                    logger.debug("Noise profile captured")
            
            # Apply noise reduction
            if self.noise_profile is not None:
                # Use spectral gating for noise reduction
                reduced_noise = nr.reduce_noise(
                    y=audio_data,
                    sr=self.sample_rate,
                    y_noise=self.noise_profile,
                    prop_decrease=self.config.noise_reduction_strength,
                    stationary=True
                )
                return reduced_noise
            else:
                # Simple noise reduction without profile
                return nr.reduce_noise(
                    y=audio_data,
                    sr=self.sample_rate,
                    prop_decrease=self.config.noise_reduction_strength,
                    stationary=True
                )
        
        except Exception as e:
            logger.warning(f"Noise cancellation failed: {e}")
            return audio_data
    
    def _apply_echo_suppression(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply simplified echo suppression.
        In production, this would use a proper AEC algorithm.
        """
        # This is a placeholder - LiveKit uses WebRTC's AEC internally
        # For now, we'll apply a simple high-pass filter to reduce low-frequency echo
        try:
            from scipy import signal
            
            # Design a high-pass filter to reduce low-frequency echo
            nyquist = self.sample_rate / 2
            cutoff = 80 / nyquist  # 80 Hz cutoff
            b, a = signal.butter(4, cutoff, btype='high')
            
            # Apply filter
            filtered = signal.filtfilt(b, a, audio_data)
            
            # Soft limit to prevent clipping
            max_val = np.max(np.abs(filtered))
            if max_val > 0.95:
                filtered = filtered * (0.95 / max_val)
            
            return filtered
        
        except ImportError:
            # scipy not available, return original
            return audio_data
        except Exception as e:
            logger.warning(f"Echo suppression failed: {e}")
            return audio_data
    
    async def detect_speech_livekit(self, audio_data: np.ndarray) -> float:
        """
        Use LiveKit VAD for speech detection.
        Returns speech probability (0.0 to 1.0).
        """
        if not self.livekit_vad or not LIVEKIT_AVAILABLE:
            return 0.0
        
        try:
            # Create AudioFrame for LiveKit VAD
            frame = AudioFrame.create(
                sample_rate=self.sample_rate,
                num_channels=1,
                samples_per_channel=len(audio_data)
            )
            
            # Copy audio data to frame
            frame_data = np.frombuffer(frame.data, dtype=np.int16)
            np.copyto(frame_data, audio_data)
            
            # Get speech probability
            speech_prob = await self.livekit_vad.detect(frame)
            self.last_speech_probability = speech_prob
            
            return speech_prob
            
        except Exception as e:
            logger.warning(f"LiveKit VAD detection failed: {e}")
            return 0.0


ReplyFnGenerator = (
    Callable[
        [tuple[int, NDArray[np.int16]], Any],
        Generator[EmitType, None, None],
    ]
    | Callable[
        [tuple[int, NDArray[np.int16]]],
        Generator[EmitType, None, None],
    ]
    | Callable[
        [tuple[int, NDArray[np.int16]]],
        AsyncGenerator[EmitType, None],
    ]
    | Callable[
        [tuple[int, NDArray[np.int16]], Any],
        AsyncGenerator[EmitType, None],
    ]
    | Callable[
        [WebRTCData],
        Generator[EmitType, None, None],
    ]
    | Callable[
        [WebRTCData, Any],
        AsyncGenerator[EmitType, None],
    ]
)


async def iterate(generator: Generator) -> Any:
    return next(generator)


class ReplyOnPause(StreamHandler):
    """
    Enhanced ReplyOnPause with LiveKit audio processing integration.
    
    This enhanced version integrates LiveKit's noise cancellation and advanced VAD
    while maintaining compatibility with FastRTC's streaming architecture.
    """

    def __init__(
        self,
        fn: ReplyFnGenerator,
        startup_fn: Callable | None = None,
        algo_options: AlgoOptions | None = None,
        model_options: ModelOptions | None = None,
        can_interrupt: bool = True,
        expected_layout: Literal["mono", "stereo"] = "mono",
        output_sample_rate: int = 24000,
        output_frame_size: int | None = None,  # Deprecated
        input_sample_rate: int = 48000,
        model: PauseDetectionModel | None = None,
        needs_args: bool = False,
    ):
        """
        Initializes the enhanced ReplyOnPause handler with LiveKit features.

        Additional LiveKit features can be configured through algo_options.livekit_config.
        """
        self.fn = fn
        self.startup_fn = startup_fn
        self.can_interrupt = can_interrupt
        self.expected_layout = expected_layout
        self.output_sample_rate = output_sample_rate
        self.input_sample_rate = input_sample_rate
        self.algo_options = algo_options or AlgoOptions()
        self.needs_args = needs_args

        if output_frame_size is not None:
            logger.warning("output_frame_size is deprecated and will be ignored")

        # Initialize original Silero model
        self.model_options = model_options or ModelOptions()
        self.model = model or get_silero_model(self.model_options)

        # Initialize LiveKit audio processor if configured
        self.livekit_processor = None
        if self.algo_options.livekit_config:
            self.livekit_processor = LiveKitAudioProcessor(
                self.algo_options.livekit_config,
                sample_rate=input_sample_rate
            )
            logger.info("LiveKit audio processing enabled")

        # State management
        self.state = AppState()
        self.generator: Generator | AsyncGenerator | None = None
        self.event = Event()
        self.loop: asyncio.AbstractEventLoop = None

        # Performance tracking
        self.total_audio_processed = 0
        self.total_speech_detected = 0

    async def initialize(self, webrtc):
        """Initialize the handler."""
        if self.startup_fn:
            if inspect.iscoroutinefunction(self.startup_fn):
                await self.startup_fn()
            else:
                self.startup_fn()
        self.loop = asyncio.get_event_loop()

    async def process_audio_chunk(
        self, audio_chunk: NDArray[np.int16], sampling_rate: int
    ) -> None:
        """
        Process a single audio chunk with LiveKit enhancements.
        """
        # Apply LiveKit audio processing if available
        if self.livekit_processor:
            processed_audio = self.livekit_processor.process_audio(audio_chunk)
        else:
            processed_audio = audio_chunk
        
        # Update state
        if self.state.stream is None:
            self.state.stream = processed_audio
            self.state.sampling_rate = sampling_rate
        else:
            self.state.stream = np.concatenate((self.state.stream, processed_audio))
        
        # Check for speech using both Silero and LiveKit VAD
        speech_detected = False
        
        # Original Silero VAD
        silero_speech = self.model(processed_audio, sampling_rate)
        
        # LiveKit VAD if available
        livekit_speech_prob = 0.0
        if self.livekit_processor and self.algo_options.livekit_config.use_livekit_vad:
            livekit_speech_prob = await self.livekit_processor.detect_speech_livekit(processed_audio)
        
        # Combine VAD results (use the more confident one)
        if self.algo_options.livekit_config and self.algo_options.livekit_config.use_livekit_vad:
            # Average the two VAD results with weight towards LiveKit
            speech_prob = (livekit_speech_prob * 0.7 + silero_speech * 0.3)
            speech_detected = speech_prob > self.algo_options.livekit_config.vad_activation_threshold
        else:
            # Use only Silero
            speech_detected = silero_speech
        
        # Update statistics
        self.total_audio_processed += len(processed_audio)
        if speech_detected:
            self.total_speech_detected += len(processed_audio)
        
        # Track speech state transitions
        audio_chunk_s = len(processed_audio) / sampling_rate
        
        if not self.state.started_talking:
            if speech_detected and audio_chunk_s >= self.algo_options.started_talking_threshold:
                self.state.started_talking = True
                logger.debug("Speech started (enhanced detection)")
        else:
            if not speech_detected or audio_chunk_s < self.algo_options.speech_threshold:
                self.state.pause_detected = True
                logger.debug("Pause detected (enhanced detection)")
    
    async def receive(self, frame: tuple[int, NDArray[np.int16]]) -> None:
        """
        Receive and process audio frame with LiveKit enhancements.
        """
        if self.state.responding:
            return

        sampling_rate, audio_chunk = frame
        
        # Convert stereo to mono if needed
        if self.expected_layout == "mono" and len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk.mean(axis=1)
        
        # Process the audio chunk
        await self.process_audio_chunk(audio_chunk, sampling_rate)
        
        # Check for pause detection or max speech duration
        current_speech_duration = len(self.state.stream) / sampling_rate if self.state.stream is not None else 0
        
        if (self.state.pause_detected and self.state.started_talking) or \
           (current_speech_duration >= self.algo_options.max_continuous_speech_s):
            
            # Signal pause detected
            self.state.responding = True
            self.event.set()
            
            # Log enhanced statistics
            if self.livekit_processor:
                logger.debug(
                    f"Enhanced pause detection - "
                    f"Speech ratio: {self.total_speech_detected/max(1, self.total_audio_processed):.2%}, "
                    f"LiveKit confidence: {self.livekit_processor.last_speech_probability:.2f}"
                )
    
    async def next(self, webrtc=None) -> AsyncGenerator[EmitType, None]:
        """
        Generate response with enhanced audio processing.
        """
        # Wait for pause detection
        await asyncio.get_event_loop().run_in_executor(None, self.event.wait)
        self.event.clear()
        
        if self.state.stream is None:
            return
        
        # Get the complete audio stream
        audio_data = (self.state.sampling_rate, self.state.stream)
        
        # Reset state for next interaction
        self.state = AppState()
        self.total_audio_processed = 0
        self.total_speech_detected = 0
        
        # Generate response
        try:
            if self.needs_args:
                args = self.latest_args if hasattr(self, 'latest_args') else []
                generator = self.fn(audio_data, *args)
            else:
                generator = self.fn(audio_data)
            
            self.generator = generator
            
            # Process generator output
            if inspect.isasyncgen(generator):
                async for item in generator:
                    if self.can_interrupt and self.state.interrupted.is_set():
                        logger.debug("Response interrupted")
                        break
                    yield item
            else:
                while True:
                    try:
                        item = await iterate(generator)
                        if self.can_interrupt and self.state.interrupted.is_set():
                            logger.debug("Response interrupted")
                            break
                        yield item
                    except StopIteration:
                        break
                        
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            raise
        finally:
            self.generator = None
            self.state.responding = False
    
    def get_statistics(self) -> dict:
        """
        Get enhanced statistics including LiveKit processing metrics.
        """
        stats = {
            "total_audio_processed_seconds": self.total_audio_processed / self.input_sample_rate,
            "total_speech_detected_seconds": self.total_speech_detected / self.input_sample_rate,
            "speech_ratio": self.total_speech_detected / max(1, self.total_audio_processed),
        }
        
        if self.livekit_processor:
            stats.update({
                "livekit_enabled": True,
                "noise_cancellation": self.algo_options.livekit_config.use_noise_cancellation,
                "livekit_vad": self.algo_options.livekit_config.use_livekit_vad,
                "last_speech_confidence": self.livekit_processor.last_speech_probability,
                "noise_profile_captured": self.livekit_processor.noise_profile is not None
            })
        else:
            stats["livekit_enabled"] = False
        
        return stats


# Maintain backward compatibility
__all__ = ['ReplyOnPause', 'AlgoOptions', 'LiveKitAudioConfig']
