"""
Enhanced ReplyOnPause with Noise Cancellation and Turn Detection
Final implementation that properly inherits from FastRTC's ReplyOnPause
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass

from fastrtc.reply_on_pause import (
    AlgoOptions,
    AppState,
    ModelOptions,
    PauseDetectionModel,
    ReplyFnGenerator,
    ReplyOnPause,
)
from fastrtc.speech_to_text import stt_for_chunks
from fastrtc.utils import audio_to_float32

# Import noise reduction module
try:
    from src.NoiseCancelation.fastrtc_noise_reduction import FastRTCNoiseReduction
except ImportError:
    FastRTCNoiseReduction = None

# Import turn detection module
try:
    from src.VAD_TurnDectection.turn_detector import LiveKitTurnDetector, TENTurnDetector
except ImportError:
    LiveKitTurnDetector = None
    TENTurnDetector = None

logger = logging.getLogger(__name__)


@dataclass
class NoiseReductionConfig:
    """Configuration for noise reduction"""
    enabled: bool = True
    model: str = "deepfilternet3"  # "deepfilternet3", "mossformer2_se_48k", "frcrn_se_16k"
    fallback_enabled: bool = True


@dataclass
class TurnDetectionConfig:
    """Configuration for semantic turn detection"""
    enabled: bool = True
    model: str = "livekit"  # "livekit" or "ten"
    confidence_threshold: float = 0.5
    fallback_to_pause: bool = True


class EnhancedReplyOnPause(ReplyOnPause):
    """
    Enhanced ReplyOnPause with noise cancellation and semantic turn detection.
    
    Inherits all FastRTC ReplyOnPause functionality while adding:
    - Noise reduction preprocessing before VAD
    - Semantic turn detection using conversation context
    - Graceful fallback mechanisms
    - Complete compatibility with FastRTC architecture
    
    Features:
    - Noise cancellation: Uses FastRTCNoiseReduction for audio preprocessing
    - Turn detection: LiveKit/TEN models for semantic conversation understanding
    - VAD integration: Works with any VAD model (HumAwareVAD, Silero, etc.)
    - Fallback support: Graceful degradation if enhanced features fail
    """
    
    def __init__(
        self,
        fn: ReplyFnGenerator,
        startup_fn=None,
        algo_options: Optional[AlgoOptions] = None,
        model_options: Optional[ModelOptions] = None,
        can_interrupt: bool = True,
        expected_layout: Literal["mono", "stereo"] = "mono",
        output_sample_rate: int = 24000,
        output_frame_size: Optional[int] = None,
        input_sample_rate: int = 48000,
        model: Optional[PauseDetectionModel] = None,
        needs_args: bool = False,
        # Enhanced parameters
        stt_model=None,
        noise_reduction_config: Optional[NoiseReductionConfig] = None,
        turn_detection_config: Optional[TurnDetectionConfig] = None,
    ):
        """
        Initialize Enhanced ReplyOnPause handler.
        
        Args:
            fn: Response generation function (required by ReplyOnPause)
            startup_fn: Optional startup function (ReplyOnPause parameter)
            algo_options: Algorithm options (ReplyOnPause parameter)
            model_options: VAD model options (ReplyOnPause parameter)
            can_interrupt: Whether responses can be interrupted (ReplyOnPause parameter)
            expected_layout: Audio channel layout (ReplyOnPause parameter)
            output_sample_rate: Output audio sample rate (ReplyOnPause parameter)
            output_frame_size: Output frame size (ReplyOnPause parameter)
            input_sample_rate: Input audio sample rate (ReplyOnPause parameter)
            model: VAD model instance (ReplyOnPause parameter)
            needs_args: Whether function needs additional args (ReplyOnPause parameter)
            
            Enhanced parameters:
            stt_model: STT model for turn detection (Enhanced parameter)
            noise_reduction_config: Noise reduction configuration (Enhanced parameter)
            turn_detection_config: Turn detection configuration (Enhanced parameter)
        """
        
        # Initialize parent class with all original parameters
        super().__init__(
            fn=fn,
            startup_fn=startup_fn,
            algo_options=algo_options,
            model_options=model_options,
            can_interrupt=can_interrupt,
            expected_layout=expected_layout,
            output_sample_rate=output_sample_rate,
            output_frame_size=output_frame_size,
            input_sample_rate=input_sample_rate,
            model=model,
            needs_args=needs_args,
        )
        
        # Enhanced configurations
        self.noise_config = noise_reduction_config or NoiseReductionConfig()
        self.turn_config = turn_detection_config or TurnDetectionConfig()
        self.stt_model = stt_model
        
        # Enhanced components
        self.noise_reducer = None
        self.turn_detector = None
        self.conversation_history = []
        
        # Initialize enhanced features
        self._init_noise_reduction()
        self._init_turn_detection()
        
        # Optimize chunk duration for semantic analysis if turn detection enabled
        if self.turn_config.enabled and self.algo_options:
            self.algo_options.audio_chunk_duration = max(
                self.algo_options.audio_chunk_duration, 3.0
            )
        
        logger.info(f"EnhancedReplyOnPause initialized:")
        logger.info(f"  - Noise reduction: {self.noise_config.enabled} ({self.noise_config.model if self.noise_config.enabled else 'disabled'})")
        logger.info(f"  - Turn detection: {self.turn_config.enabled} ({self.turn_config.model if self.turn_config.enabled else 'disabled'})")
        logger.info(f"  - STT model: {'provided' if self.stt_model else 'none'}")
    
    def _init_noise_reduction(self):
        """Initialize noise reduction model"""
        if not self.noise_config.enabled:
            logger.info("Noise reduction disabled")
            return
            
        if FastRTCNoiseReduction is None:
            logger.warning("FastRTCNoiseReduction not available, disabling noise reduction")
            self.noise_config.enabled = False
            return
            
        try:
            self.noise_reducer = FastRTCNoiseReduction(
                preferred_model=self.noise_config.model
            )
            
            model_info = self.noise_reducer.get_model_info()
            logger.info(f"Noise reduction initialized: {model_info}")
            
        except Exception as e:
            logger.error(f"Failed to initialize noise reduction model '{self.noise_config.model}': {e}")
            
            if self.noise_config.fallback_enabled:
                logger.info("Disabling noise reduction, continuing with original audio")
                self.noise_config.enabled = False
                self.noise_reducer = None
            else:
                raise RuntimeError(f"Noise reduction initialization failed: {e}")
    
    def _init_turn_detection(self):
        """Initialize semantic turn detection model"""
        if not self.turn_config.enabled:
            logger.info("Turn detection disabled")
            return
            
        if LiveKitTurnDetector is None and TENTurnDetector is None:
            logger.warning("Turn detection modules not available, disabling turn detection")
            self.turn_config.enabled = False
            return
            
        try:
            if self.turn_config.model == "livekit" and LiveKitTurnDetector is not None:
                self.turn_detector = LiveKitTurnDetector(language="multilingual")
                logger.info("LiveKit turn detector initialized (multilingual)")
            elif self.turn_config.model == "ten" and TENTurnDetector is not None:
                self.turn_detector = TENTurnDetector()
                logger.info("TEN turn detector initialized")
            else:
                logger.warning(f"Turn detection model '{self.turn_config.model}' not available")
                if self.turn_config.fallback_to_pause:
                    logger.info("Falling back to pause-based turn detection")
                    self.turn_config.enabled = False
                else:
                    raise RuntimeError(f"Turn detection model '{self.turn_config.model}' not available")
                    
        except Exception as e:
            logger.error(f"Failed to initialize turn detection: {e}")
            
            if self.turn_config.fallback_to_pause:
                logger.info("Disabling semantic turn detection, falling back to pause detection")
                self.turn_config.enabled = False
                self.turn_detector = None
            else:
                raise RuntimeError(f"Turn detection initialization failed: {e}")
    
    def _preprocess_audio(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Apply noise reduction preprocessing to audio.
        
        Args:
            audio: Raw audio array
            sampling_rate: Audio sampling rate
            
        Returns:
            Processed audio array (same shape as input)
        """
        if not self.noise_config.enabled or self.noise_reducer is None:
            return audio
            
        try:
            # Apply noise reduction
            enhanced_audio = self.noise_reducer.process_audio_chunk(
                audio, sampling_rate
            )
            
            logger.debug(f"Applied noise reduction to {len(audio)} samples")
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Noise reduction processing failed: {e}")
            
            if self.noise_config.fallback_enabled:
                logger.warning("Falling back to original audio")
                return audio
            else:
                raise
    
    def _detect_semantic_turn_end(self, text: str) -> bool:
        """
        Use semantic turn detection to determine if turn should end.
        
        Args:
            text: Transcribed text from current audio
            
        Returns:
            bool: True if turn should end based on semantic analysis
        """
        if not self.turn_config.enabled or not self.turn_detector or not text.strip():
            return False
            
        try:
            # Prepare conversation context - use recent history + current message
            messages = self.conversation_history[-3:] + [{"role": "user", "content": text.strip()}]
            
            # Get turn detection prediction using the correct method
            result = self.turn_detector.should_end_turn(messages)
            
            should_end = result.get("should_end", False)
            confidence = result.get("eou_probability", 0.0)
            
            logger.debug(f"Semantic turn detection - Should end: {should_end}, Confidence: {confidence:.3f}")
            
            if should_end:
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": text.strip()})
                # Keep history manageable
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-8:]
            
            return should_end
            
        except Exception as e:
            logger.error(f"Semantic turn detection failed: {e}")
            return False
    
    def determine_pause(self, audio: np.ndarray, sampling_rate: int, state: AppState) -> bool:
        """
        Enhanced pause detection with noise reduction and semantic turn detection.
        
        This is the main integration point where we inject our enhancements
        while maintaining compatibility with FastRTC's architecture.
        
        Args:
            audio: Raw audio chunk
            sampling_rate: Audio sampling rate
            state: Current application state
            
        Returns:
            bool: True if pause/turn end detected
        """
        
        # Step 1: Apply noise reduction preprocessing
        processed_audio = self._preprocess_audio(audio, sampling_rate)
        
        # Step 2: Check for semantic turn detection if enabled
        if self.turn_config.enabled and self.stt_model:
            duration = len(processed_audio) / sampling_rate
            
            if duration >= self.algo_options.audio_chunk_duration:
                try:
                    # Convert audio for STT processing
                    import librosa
                    
                    audio_f32 = audio_to_float32((sampling_rate, processed_audio))
                    audio_rs = librosa.resample(
                        audio_f32, orig_sr=sampling_rate, target_sr=16000
                    )
                    
                    # Get VAD chunks and transcription
                    _, chunks = self.model.vad((16000, audio_rs), self.model_options)
                    text = stt_for_chunks(self.stt_model, (16000, audio_rs), chunks)
                    
                    # Use semantic turn detection
                    if self._detect_semantic_turn_end(text):
                        state.buffer = None
                        state.stream = processed_audio
                        logger.debug(f"Semantic turn end detected: '{text}'")
                        return True
                    
                    state.stream = None
                    return False
                    
                except Exception as e:
                    logger.error(f"Semantic turn detection processing failed: {e}")
                    # Fall through to default pause detection
        
        # Step 3: Fall back to default pause detection with processed audio
        return super().determine_pause(processed_audio, sampling_rate, state)
    
    def add_assistant_response(self, response_text: str):
        """
        Add assistant response to conversation history for context.
        
        Args:
            response_text: The assistant's response text
        """
        if self.turn_config.enabled and response_text.strip():
            assistant_message = {"role": "assistant", "content": response_text.strip()}
            self.conversation_history.append(assistant_message)
            
            # Keep history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-8:]
            
            logger.debug(f"Added assistant response to conversation history: '{response_text[:50]}...'")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information about the enhanced handler.
        
        Returns:
            Dict containing status of all components
        """
        status = {
            "enhanced_reply_on_pause": True,
            "version": "final",
            "noise_reduction": {
                "enabled": self.noise_config.enabled,
                "model": self.noise_config.model if self.noise_config.enabled else None,
                "status": "ready" if self.noise_reducer else "disabled"
            },
            "turn_detection": {
                "enabled": self.turn_config.enabled,
                "model": self.turn_config.model if self.turn_config.enabled else None,
                "status": "ready" if self.turn_detector else "disabled",
                "confidence_threshold": self.turn_config.confidence_threshold
            },
            "vad": {
                "model": type(self.model).__name__ if self.model else "default_silero",
                "status": "active"
            },
            "stt": {
                "enabled": self.turn_config.enabled and self.stt_model is not None,
                "status": "ready" if self.stt_model else "not_provided"
            },
            "conversation_history_length": len(self.conversation_history),
            "audio_processing": {
                "chunk_duration": self.algo_options.audio_chunk_duration,
                "input_sample_rate": self.input_sample_rate,
                "output_sample_rate": self.output_sample_rate
            }
        }
        
        # Add model info if available
        if self.noise_reducer:
            try:
                model_info = self.noise_reducer.get_model_info()
                status["noise_reduction"]["model_info"] = model_info
            except Exception:
                pass
                
        return status
    
    def update_configs(self, 
                      noise_config: Optional[NoiseReductionConfig] = None,
                      turn_config: Optional[TurnDetectionConfig] = None):
        """
        Update configurations at runtime.
        
        Args:
            noise_config: New noise reduction configuration
            turn_config: New turn detection configuration
        """
        if noise_config:
            old_enabled = self.noise_config.enabled
            self.noise_config = noise_config
            
            # Reinitialize if enabling noise reduction
            if noise_config.enabled and not old_enabled:
                self._init_noise_reduction()
            elif not noise_config.enabled:
                self.noise_reducer = None
                
            logger.info(f"Noise reduction config updated: enabled={noise_config.enabled}, model={noise_config.model}")
        
        if turn_config:
            old_enabled = self.turn_config.enabled
            self.turn_config = turn_config
            
            # Reinitialize if enabling turn detection
            if turn_config.enabled and not old_enabled:
                self._init_turn_detection()
            elif not turn_config.enabled:
                self.turn_detector = None
                
            logger.info(f"Turn detection config updated: enabled={turn_config.enabled}, model={turn_config.model}")
    
    def copy(self):
        """
        Create a copy of the handler (required by FastRTC).
        
        Returns:
            New EnhancedReplyOnPause instance with same configuration
        """
        return EnhancedReplyOnPause(
            fn=self.fn,
            startup_fn=self.startup_fn,
            algo_options=self.algo_options,
            model_options=self.model_options,
            can_interrupt=self.can_interrupt,
            expected_layout=self.expected_layout,
            output_sample_rate=self.output_sample_rate,
            output_frame_size=self.output_frame_size,
            input_sample_rate=self.input_sample_rate,
            model=self.model,
            needs_args=self.needs_args,
            stt_model=self.stt_model,
            noise_reduction_config=self.noise_config,
            turn_detection_config=self.turn_config,
        )
    
    def reset(self):
        """Reset handler state (extends parent method)"""
        super().reset()
        self.conversation_history = []
        logger.debug("Enhanced handler state reset - conversation history cleared")


# Convenience functions for easy instantiation
def create_enhanced_handler(
    response_fn,
    stt_model=None,
    enable_noise_reduction: bool = True,
    noise_model: str = "deepfilternet3",
    enable_turn_detection: bool = True,
    turn_model: str = "livekit",
    **kwargs
):
    """
    Create an EnhancedReplyOnPause handler with sensible defaults.
    
    Args:
        response_fn: Function to generate responses
        stt_model: STT model for turn detection
        enable_noise_reduction: Whether to enable noise reduction
        noise_model: Noise reduction model ("deepfilternet3", "mossformer2_se_48k", "frcrn_se_16k")
        enable_turn_detection: Whether to enable semantic turn detection
        turn_model: Turn detection model ("livekit" or "ten")
        **kwargs: Additional arguments for ReplyOnPause
        
    Returns:
        Configured EnhancedReplyOnPause instance
    """
    
    noise_config = NoiseReductionConfig(
        enabled=enable_noise_reduction,
        model=noise_model,
        fallback_enabled=True
    ) if enable_noise_reduction else NoiseReductionConfig(enabled=False)
    
    turn_config = TurnDetectionConfig(
        enabled=enable_turn_detection,
        model=turn_model,
        confidence_threshold=0.5,
        fallback_to_pause=True
    ) if enable_turn_detection else TurnDetectionConfig(enabled=False)
    
    return EnhancedReplyOnPause(
        fn=response_fn,
        stt_model=stt_model,
        noise_reduction_config=noise_config,
        turn_detection_config=turn_config,
        **kwargs
    )


def create_basic_enhanced_handler(response_fn, stt_model=None, **kwargs):
    """
    Create a basic enhanced handler with all features enabled.
    
    Args:
        response_fn: Function to generate responses
        stt_model: STT model for turn detection
        **kwargs: Additional arguments for ReplyOnPause
        
    Returns:
        EnhancedReplyOnPause instance with all features enabled
    """
    return create_enhanced_handler(
        response_fn=response_fn,
        stt_model=stt_model,
        enable_noise_reduction=True,
        noise_model="deepfilternet3",
        enable_turn_detection=True,
        turn_model="livekit",
        **kwargs
    )


# # Example usage demonstration
# if __name__ == "__main__":
#     def example_response_function(audio_data):
#         """Example response function for testing"""
#         yield "Enhanced response with noise reduction and turn detection"
    
#     # Create handler with all features
#     handler = create_enhanced_handler(
#         response_fn=example_response_function,
#         stt_model=None,  # Provide your STT model here
#         enable_noise_reduction=True,
#         noise_model="deepfilternet3",
#         enable_turn_detection=True,
#         turn_model="livekit"
#     )
    
#     print("Handler Status:")
#     status = handler.get_status()
#     for key, value in status.items():
#         print(f"  {key}: {value}")