"""
Enhanced ReplyOnPause with Noise Cancellation and Turn Detection
Lean and efficient implementation extending FastRTC's ReplyOnPause
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, Literal, List
from dataclasses import dataclass, field

# FastRTC imports
from fastrtc.pause_detection import ModelOptions, PauseDetectionModel
from fastrtc.reply_on_pause import AlgoOptions, AppState, ReplyFnGenerator, ReplyOnPause
from fastrtc.utils import create_message

logger = logging.getLogger(__name__)


@dataclass
class EnhancedAppState(AppState):
    """Extended state with enhancement tracking"""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    last_transcript: str = ""
    noise_reduction_enabled: bool = True
    turn_detection_enabled: bool = True


@dataclass 
class EnhancementConfig:
    """Lean configuration for all enhancements"""
    # Noise reduction
    noise_model: str = "deepfilternet3"
    noise_enabled: bool = True
    
    # Turn detection  
    turn_model: str = "livekit"  # "livekit" or "ten"
    turn_enabled: bool = True
    turn_threshold: float = 0.5
    
    # STT for turn detection
    stt_model: Optional[Any] = None
    
    # Fallback behavior
    fallback_on_error: bool = True


class EnhancedReplyOnPause(ReplyOnPause):
    """
    Enhanced ReplyOnPause with noise cancellation and semantic turn detection.
    
    Adds:
    - Real-time noise cancellation before VAD processing
    - Semantic turn detection using conversation context
    - Graceful fallback to standard pause detection
    - Minimal performance overhead
    """
    
    def __init__(
        self,
        fn: ReplyFnGenerator,
        stt_model=None,
        noise_reduction_config=None,
        turn_detection_config=None,
        **kwargs  # All original ReplyOnPause parameters
    ):
        """Initialize with enhancement capabilities"""
        # Extract enhancement-specific parameters before calling super()
        self._stt_model = stt_model
        
        # Call parent constructor with only valid ReplyOnPause parameters
        super().__init__(fn, **kwargs)
        
        # Create unified config from separate configs for backwards compatibility
        self.config = EnhancementConfig()
        
        if noise_reduction_config:
            self.config.noise_enabled = noise_reduction_config.enabled
            self.config.noise_model = noise_reduction_config.model
            self.config.fallback_on_error = noise_reduction_config.fallback_enabled
        
        if turn_detection_config:
            self.config.turn_enabled = turn_detection_config.enabled
            self.config.turn_model = turn_detection_config.model
            self.config.turn_threshold = turn_detection_config.confidence_threshold
            self.config.fallback_on_error = turn_detection_config.fallback_to_pause
            
        self.config.stt_model = self._stt_model
        
        self.enhanced_state = EnhancedAppState()
        
        # Initialize components lazily for efficiency
        self._noise_reducer = None
        self._turn_detector = None
    
    @property
    def noise_reducer(self):
        """Lazy initialization of noise reducer"""
        if self._noise_reducer is None and self.config.noise_enabled:
            try:
                from src.NoiseCancelation.fastrtc_noise_reduction import FastRTCNoiseReduction
                self._noise_reducer = FastRTCNoiseReduction(self.config.noise_model)
                logger.debug(f"Noise reducer loaded: {self.config.noise_model}")
            except Exception as e:
                logger.warning(f"Noise reducer failed to load: {e}")
                if not self.config.fallback_on_error:
                    raise
        return self._noise_reducer
    
    @property 
    def turn_detector(self):
        """Lazy initialization of turn detector"""
        if self._turn_detector is None and self.config.turn_enabled:
            try:
                if self.config.turn_model == "livekit":
                    from src.VAD_TurnDectection.turn_detector import LiveKitTurnDetector
                    self._turn_detector = LiveKitTurnDetector()
                else:
                    from src.VAD_TurnDectection.turn_detector import TENTurnDetector
                    self._turn_detector = TENTurnDetector()
                logger.debug(f"Turn detector loaded: {self.config.turn_model}")
            except Exception as e:
                logger.warning(f"Turn detector failed to load: {e}")
                if not self.config.fallback_on_error:
                    raise
        return self._turn_detector
    
    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """Enhanced receive with noise cancellation preprocessing"""
        if self.state.responding and not self.can_interrupt:
            return
        
        # Apply noise cancellation if enabled
        enhanced_frame = self._apply_noise_reduction(frame)
        
        # Process audio with enhanced frame
        self.process_audio(enhanced_frame, self.state)
        
        if self.state.pause_detected:
            self.event.set()
            if self.can_interrupt and self.state.responding:
                self._close_generator()
                self.generator = None
            if self.can_interrupt:
                self.clear_queue()
    
    def _apply_noise_reduction(self, frame: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
        """Apply noise reduction to audio frame"""
        if not self.config.noise_enabled or self.noise_reducer is None:
            return frame
        
        try:
            sample_rate, audio = frame
            
            # Skip if audio too short
            if len(audio) < 480:  # ~10ms at 48kHz
                return frame
            
            # Apply noise reduction
            enhanced_audio = self.noise_reducer.process_audio_chunk(audio, sample_rate)
            
            # Ensure same dtype and shape
            enhanced_audio = enhanced_audio.astype(audio.dtype)
            if enhanced_audio.shape != audio.shape:
                enhanced_audio = enhanced_audio.reshape(audio.shape)
            
            return (sample_rate, enhanced_audio)
            
        except Exception as e:
            logger.debug(f"Noise reduction failed: {e}")
            return frame  # Fallback to original
    
    def determine_pause(self, audio: np.ndarray, sampling_rate: int, state: AppState) -> bool:
        """Enhanced pause detection with semantic turn analysis"""
        # First run standard VAD-based pause detection
        vad_pause = super().determine_pause(audio, sampling_rate, state)
        
        # If VAD doesn't detect pause, return False
        if not vad_pause:
            return False
        
        # If turn detection disabled, return VAD result
        if not self.config.turn_enabled or self.turn_detector is None:
            return vad_pause
        
        # Apply semantic turn detection
        return self._check_semantic_turn_end(state)
    
    def _check_semantic_turn_end(self, state: AppState) -> bool:
        """Check if conversation semantically indicates turn end"""
        try:
            # Get transcript of current speech
            if state.stream is None or len(state.stream) == 0:
                return True  # Empty speech, end turn
            
            transcript = self._transcribe_audio(state.stream, state.sampling_rate)
            if not transcript.strip():
                return True  # No speech detected
            
            # Update conversation history
            self.enhanced_state.last_transcript = transcript
            self._update_conversation_history("user", transcript)
            
            # Check turn end probability
            result = self.turn_detector.should_end_turn(self.enhanced_state.conversation_history)
            
            should_end = result.get("should_end", True)
            confidence = result.get("eou_probability", 0.5)
            
            logger.debug(f"Turn detection: should_end={should_end}, confidence={confidence:.3f}")
            
            return should_end and confidence >= self.config.turn_threshold
            
        except Exception as e:
            logger.debug(f"Semantic turn detection failed: {e}")
            return True  # Fallback to ending turn
    
    def _transcribe_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio to text for turn detection"""
        if self._stt_model is None:
            return ""  # No STT available
        
        try:
            # Convert to expected format
            audio_float = audio.astype(np.float32) / 32768.0
            
            # Transcribe using provided STT model
            if hasattr(self._stt_model, 'transcribe'):
                return self._stt_model.transcribe(audio_float, sample_rate)
            elif callable(self._stt_model):
                return self._stt_model((sample_rate, audio))
            else:
                return ""
                
        except Exception as e:
            logger.debug(f"Transcription failed: {e}")
            return ""
    
    def _update_conversation_history(self, role: str, content: str):
        """Update conversation history for turn detection"""
        if not content.strip():
            return
        
        # Keep last 10 messages for efficiency
        self.enhanced_state.conversation_history.append({"role": role, "content": content})
        if len(self.enhanced_state.conversation_history) > 10:
            self.enhanced_state.conversation_history.pop(0)
    
    def reset(self):
        """Enhanced reset with conversation state"""
        super().reset()
        self.enhanced_state = EnhancedAppState()
        logger.debug("Enhanced state reset")
    
    def copy(self):
        """Create enhanced copy"""
        return EnhancedReplyOnPause(
            fn=self.fn,
            enhancement_config=self.config,
            startup_fn=self.startup_fn,
            algo_options=self.algo_options,
            model_options=self.model_options,
            can_interrupt=self.can_interrupt,
            expected_layout=self.expected_layout,
            output_sample_rate=self.output_sample_rate,
            input_sample_rate=self.input_sample_rate,
            model=self.model,
            needs_args=self.needs_args
        )
    
    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get enhancement performance statistics"""
        return {
            "noise_reduction": {
                "enabled": self.config.noise_enabled,
                "model": self.config.noise_model,
                "loaded": self._noise_reducer is not None
            },
            "turn_detection": {
                "enabled": self.config.turn_enabled, 
                "model": self.config.turn_model,
                "loaded": self._turn_detector is not None,
                "threshold": self.config.turn_threshold
            },
            "conversation": {
                "history_length": len(self.enhanced_state.conversation_history),
                "last_transcript": self.enhanced_state.last_transcript
            }
        }


# Convenience factory functions
def create_enhanced_reply_on_pause(
    fn: ReplyFnGenerator,
    stt_model=None,
    noise_model: str = "deepfilternet3",
    turn_model: str = "livekit",
    **kwargs
) -> EnhancedReplyOnPause:
    """Factory function for easy enhanced handler creation"""
    config = EnhancementConfig(
        noise_model=noise_model,
        turn_model=turn_model,
        stt_model=stt_model
    )
    return EnhancedReplyOnPause(fn, config, **kwargs)


def create_noise_only_reply_on_pause(
    fn: ReplyFnGenerator,
    noise_model: str = "deepfilternet3",
    **kwargs
) -> EnhancedReplyOnPause:
    """Factory for noise reduction only"""
    config = EnhancementConfig(
        noise_model=noise_model,
        noise_enabled=True,
        turn_enabled=False
    )
    return EnhancedReplyOnPause(fn, config, **kwargs)


def create_turn_only_reply_on_pause(
    fn: ReplyFnGenerator,
    stt_model,
    turn_model: str = "livekit", 
    **kwargs
) -> EnhancedReplyOnPause:
    """Factory for turn detection only"""
    config = EnhancementConfig(
        noise_enabled=False,
        turn_model=turn_model,
        turn_enabled=True,
        stt_model=stt_model
    )
    return EnhancedReplyOnPause(fn, config, **kwargs)