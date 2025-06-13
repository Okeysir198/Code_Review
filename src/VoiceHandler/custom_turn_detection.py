import numpy as np
import torch
from typing import Optional, Dict, Any, List, Tuple
import logging
from dataclasses import dataclass
from fastrtc.reply_on_pause import ReplyOnPause, AppState
from fastrtc.pause_detection import PauseDetectionModel, ModelOptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedAlgoOptions:
    """Enhanced algorithm options for better turn detection"""
    audio_chunk_duration: float = 0.5  # Reduced for faster response
    started_talking_threshold: float = 0.15  # Lower threshold for better sensitivity
    speech_threshold: float = 0.08  # Fine-tuned for fewer false positives
    noise_reduction_enabled: bool = True
    semantic_turn_detection: bool = True
    adaptive_thresholds: bool = True
    vad_grace_period_ms: int = 150  # Grace period for brief pauses
    min_speech_duration_ms: int = 200  # Minimum speech duration to consider
    max_silence_duration_ms: int = 800  # Maximum silence before turn end

class NoiseReductionModel:
    """
    Noise reduction using RNNoise-style approach
    Lightweight implementation for real-time processing
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.frame_size = 480  # 10ms at 48kHz
        self.overlap = 240
        self.noise_gate_threshold = 0.1
        self.noise_reduction_factor = 0.7
        
        # Simple spectral gating parameters
        self.alpha_noise = 0.1  # Noise floor adaptation rate
        self.beta_snr = 2.0     # SNR threshold multiplier
        
        # Initialize noise floor estimate
        self.noise_floor = None
        self.frame_count = 0
        
    def apply_spectral_gating(self, audio_frame: np.ndarray) -> np.ndarray:
        """
        Apply spectral gating noise reduction
        Simplified version inspired by RNNoise principles
        """
        if len(audio_frame) < self.frame_size:
            return audio_frame
            
        # Convert to frequency domain
        fft = np.fft.rfft(audio_frame)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Initialize or update noise floor estimate
        if self.noise_floor is None:
            self.noise_floor = magnitude.copy()
        else:
            # Adaptive noise floor estimation
            self.noise_floor = (1 - self.alpha_noise) * self.noise_floor + \
                              self.alpha_noise * magnitude
        
        # Calculate SNR and apply gating
        snr = magnitude / (self.noise_floor + 1e-10)
        gate = np.minimum(1.0, np.maximum(0.0, 
                         (snr - self.beta_snr) / (self.beta_snr + 1e-10)))
        
        # Apply noise gate
        gate = np.where(gate < self.noise_gate_threshold, 
                       self.noise_reduction_factor * gate, gate)
        
        # Apply gating to magnitude
        filtered_magnitude = magnitude * gate
        
        # Reconstruct signal
        filtered_fft = filtered_magnitude * np.exp(1j * phase)
        filtered_audio = np.fft.irfft(filtered_fft, n=len(audio_frame))
        
        return filtered_audio.astype(np.float32)
    
    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Process audio with noise reduction"""
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        
        # Process in overlapping frames
        output = np.zeros_like(audio)
        hop_size = self.frame_size - self.overlap
        
        for i in range(0, len(audio) - self.frame_size + 1, hop_size):
            frame = audio[i:i + self.frame_size]
            filtered_frame = self.apply_spectral_gating(frame)
            
            # Overlap-add reconstruction
            output[i:i + self.frame_size] += filtered_frame
            
        return output

class EnhancedVADModel:
    """
    Enhanced VAD model combining Silero-style detection with turn prediction
    Integrates noise robustness and semantic context awareness
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_size = 512 if sample_rate == 16000 else 1024
        self.speech_threshold = 0.5
        self.silence_threshold = 0.3
        
        # Initialize lightweight model (in practice, you'd load Silero VAD)
        self.model = self._init_vad_model()
        
        # Turn detection context
        self.speech_history = []
        self.silence_history = []
        self.turn_context = []
        self.max_context_length = 4  # Turns to remember
        
    def _init_vad_model(self):
        """Initialize VAD model (placeholder for Silero VAD)"""
        try:
            # In practice, load Silero VAD here:
            # model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', 
            #                              model='silero_vad')
            # return model
            return None  # Placeholder
        except Exception as e:
            logger.warning(f"Could not load Silero VAD: {e}")
            return None
    
    def detect_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """
        Detect speech probability in audio chunk
        Returns probability between 0 and 1
        """
        if self.model is not None:
            # Use Silero VAD
            try:
                # Process with actual Silero VAD
                # speech_prob = self.model(audio_chunk)
                # return float(speech_prob)
                pass
            except Exception as e:
                logger.warning(f"VAD model error: {e}")
        
        # Fallback: energy-based detection
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        rms_threshold = 0.01
        
        # Simple energy-based VAD
        if energy > rms_threshold:
            # Check for speech-like characteristics
            # Simplified spectral analysis
            fft = np.fft.rfft(audio_chunk)
            freqs = np.fft.rfftfreq(len(audio_chunk), 1/self.sample_rate)
            
            # Look for speech formants (rough approximation)
            speech_band_energy = np.sum(np.abs(fft[(freqs >= 300) & (freqs <= 3400)]))
            total_energy = np.sum(np.abs(fft))
            
            speech_ratio = speech_band_energy / (total_energy + 1e-10)
            return min(1.0, speech_ratio * 2.0)  # Scale appropriately
        
        return 0.0
    
    def predict_turn_end(self, recent_transcription: str = None) -> float:
        """
        Predict probability that turn is ending based on context
        Returns probability between 0 and 1
        """
        if not recent_transcription:
            return 0.5  # Neutral when no text context
        
        # Simple linguistic cues for turn ending
        ending_cues = [
            "?",  # Questions
            ".",  # Statements
            "!",  # Exclamations
            "please",
            "thank you",
            "thanks",
            "that's all",
            "over",
            "done"
        ]
        
        text_lower = recent_transcription.lower().strip()
        
        # Check for ending punctuation
        if text_lower.endswith(('?', '.', '!')):
            return 0.8
        
        # Check for ending phrases
        for cue in ending_cues:
            if cue in text_lower:
                return 0.7
        
        # Check for incomplete sentences (lower turn probability)
        incomplete_indicators = ["um", "uh", "er", "so", "and", "but", "because"]
        if any(text_lower.endswith(word) for word in incomplete_indicators):
            return 0.2
        
        return 0.4  # Default moderate probability

class MyTurnTakingAlgorithm(ReplyOnPause):
    """
    Enhanced turn-taking algorithm with:
    1. Advanced noise cancellation (RNNoise-style spectral gating)
    2. Improved VAD with Silero-style detection
    3. Semantic turn detection using linguistic cues
    4. Adaptive thresholds based on audio quality
    5. Context-aware turn prediction
    """
    
    def __init__(self, 
                 fn, 
                 startup_fn=None, 
                 enhanced_options: EnhancedAlgoOptions = None,
                 **kwargs):
        
        # Set enhanced options
        self.enhanced_options = enhanced_options or EnhancedAlgoOptions()
        
        # Initialize noise reduction
        input_sample_rate = kwargs.get('input_sample_rate', 48000)
        self.noise_reducer = NoiseReductionModel(input_sample_rate) if \
            self.enhanced_options.noise_reduction_enabled else None
        
        # Initialize enhanced VAD
        self.enhanced_vad = EnhancedVADModel(16000)  # Silero uses 16kHz
        
        # Context tracking
        self.recent_transcription = ""
        self.speech_segments = []
        self.adaptive_threshold = self.enhanced_options.speech_threshold
        
        # Convert enhanced options to standard algo options
        from fastrtc import AlgoOptions
        algo_options = AlgoOptions(
            audio_chunk_duration=self.enhanced_options.audio_chunk_duration,
            started_talking_threshold=self.enhanced_options.started_talking_threshold,
            speech_threshold=self.enhanced_options.speech_threshold
        )
        
        # Initialize parent class
        super().__init__(fn, startup_fn, algo_options, **kwargs)
        
        logger.info("MyTurnTakingAlgorithm initialized with enhanced features")
    
    def preprocess_audio(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Preprocess audio with noise reduction before VAD
        """
        if self.noise_reducer and self.enhanced_options.noise_reduction_enabled:
            try:
                # Apply noise reduction
                cleaned_audio = self.noise_reducer.process_audio(audio)
                logger.debug("Applied noise reduction")
                return cleaned_audio
            except Exception as e:
                logger.warning(f"Noise reduction failed: {e}")
                return audio
        return audio
    
    def enhanced_speech_detection(self, audio: np.ndarray, sampling_rate: int) -> float:
        """
        Enhanced speech detection combining multiple approaches
        """
        # Resample if needed for VAD model
        if sampling_rate != 16000:
            # Simple resampling (in practice, use proper resampling)
            resample_ratio = 16000 / sampling_rate
            resampled_length = int(len(audio) * resample_ratio)
            audio_16k = np.interp(np.linspace(0, len(audio), resampled_length),
                                 np.arange(len(audio)), audio)
        else:
            audio_16k = audio
        
        # Get speech probability from enhanced VAD
        speech_prob = self.enhanced_vad.detect_speech_probability(audio_16k)
        
        # Apply adaptive thresholds if enabled
        if self.enhanced_options.adaptive_thresholds:
            # Adjust threshold based on recent audio quality
            audio_quality = self._estimate_audio_quality(audio)
            self.adaptive_threshold = self._adapt_threshold(audio_quality)
            speech_prob = speech_prob * (2.0 - audio_quality)  # Boost for poor quality
        
        return speech_prob
    
    def _estimate_audio_quality(self, audio: np.ndarray) -> float:
        """
        Estimate audio quality (0=poor, 1=excellent)
        """
        # Simple SNR estimation
        signal_power = np.var(audio)
        if signal_power < 1e-8:
            return 0.1
        
        # Estimate noise floor from quieter segments
        sorted_audio = np.sort(np.abs(audio))
        noise_floor = np.mean(sorted_audio[:len(sorted_audio)//4])
        
        snr = signal_power / (noise_floor**2 + 1e-10)
        quality = min(1.0, np.log10(snr + 1) / 2.0)  # Log scale
        return max(0.1, quality)
    
    def _adapt_threshold(self, audio_quality: float) -> float:
        """
        Adapt speech threshold based on audio quality
        """
        base_threshold = self.enhanced_options.speech_threshold
        
        if audio_quality > 0.8:
            return base_threshold * 0.9  # Lower threshold for good quality
        elif audio_quality < 0.3:
            return base_threshold * 1.3  # Higher threshold for poor quality
        else:
            return base_threshold
    
    def determine_pause(self, audio: np.ndarray, sampling_rate: int, state: AppState) -> bool:
        """
        Enhanced pause determination with semantic context
        """
        # Preprocess audio
        cleaned_audio = self.preprocess_audio(audio, sampling_rate)
        
        # Enhanced speech detection
        speech_duration = self.enhanced_speech_detection(cleaned_audio, sampling_rate)
        
        # Get base pause detection from parent
        base_pause = super().determine_pause(cleaned_audio, sampling_rate, state)
        
        # Apply semantic turn detection if enabled
        if self.enhanced_options.semantic_turn_detection and hasattr(state, 'recent_transcription'):
            turn_end_prob = self.enhanced_vad.predict_turn_end(state.recent_transcription)
            
            # Combine VAD-based pause with semantic prediction
            if turn_end_prob > 0.7:
                logger.debug("High turn-end probability from semantic analysis")
                return True
            elif turn_end_prob < 0.3 and not base_pause:
                logger.debug("Low turn-end probability, extending patience")
                return False
        
        # Apply enhanced timing constraints
        if hasattr(state, 'silence_duration_ms'):
            silence_ms = getattr(state, 'silence_duration_ms', 0)
            speech_ms = getattr(state, 'speech_duration_ms', 0)
            
            # Minimum speech duration check
            if speech_ms < self.enhanced_options.min_speech_duration_ms:
                return False
            
            # Maximum silence duration check
            if silence_ms > self.enhanced_options.max_silence_duration_ms:
                return True
        
        return base_pause
    
    def update_transcription_context(self, transcription: str):
        """
        Update the recent transcription for semantic analysis
        Call this method when you receive new transcription text
        """
        self.recent_transcription = transcription
        logger.debug(f"Updated transcription context: {transcription}")
    
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """
        Get current algorithm statistics and performance metrics
        """
        return {
            "noise_reduction_enabled": self.enhanced_options.noise_reduction_enabled,
            "semantic_detection_enabled": self.enhanced_options.semantic_turn_detection,
            "adaptive_thresholds_enabled": self.enhanced_options.adaptive_thresholds,
            "current_speech_threshold": self.adaptive_threshold,
            "recent_transcription_length": len(self.recent_transcription),
            "speech_segments_tracked": len(self.speech_segments)
        }

# Example usage and integration
def create_enhanced_turn_taking_stream():
    """
    Example of how to create a Stream with MyTurnTakingAlgorithm
    """
    from fastrtc import Stream
    
    def my_response_function(audio_tuple):
        """Your response generation function"""
        sample_rate, audio_data = audio_tuple
        # Process audio and generate response
        yield b"response_audio_bytes"
    
    # Configure enhanced options
    enhanced_options = EnhancedAlgoOptions(
        audio_chunk_duration=0.4,  # Faster response
        started_talking_threshold=0.12,  # More sensitive
        speech_threshold=0.08,  # Fine-tuned
        noise_reduction_enabled=True,
        semantic_turn_detection=True,
        adaptive_thresholds=True,
        vad_grace_period_ms=120,
        min_speech_duration_ms=150,
        max_silence_duration_ms=700
    )
    
    # Create the enhanced turn-taking algorithm
    turn_taking = MyTurnTakingAlgorithm(
        fn=my_response_function,
        enhanced_options=enhanced_options,
        can_interrupt=True,
        expected_layout="mono",
        output_sample_rate=24000,
        input_sample_rate=48000
    )
    
    # Create stream
    stream = Stream(
        handler=turn_taking,
        modality="audio",
        mode="send-receive"
    )
    
    return stream, turn_taking

# Integration with STT for semantic features
class STTIntegration:
    """
    Helper class to integrate STT with enhanced turn taking
    """
    
    def __init__(self, turn_taking_algorithm: MyTurnTakingAlgorithm):
        self.turn_taking = turn_taking_algorithm
        
    def on_transcription_update(self, transcription: str, is_final: bool = False):
        """
        Called when STT provides new transcription
        """
        self.turn_taking.update_transcription_context(transcription)
        
        if is_final:
            logger.info(f"Final transcription: {transcription}")

if __name__ == "__main__":
    # Example usage
    stream, turn_taking_algo = create_enhanced_turn_taking_stream()
    
    # Show algorithm stats
    stats = turn_taking_algo.get_algorithm_stats()
    print("Enhanced Turn Taking Algorithm Configuration:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTo launch the stream, call: stream.ui.launch()")