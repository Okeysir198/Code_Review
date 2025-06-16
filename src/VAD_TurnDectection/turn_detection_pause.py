"""
Lean Turn Detection Pause Detector
Integrates LiveKitTurnDetector or TENTurnDetector with FastRTC's ReplyOnPause
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from fastrtc.reply_on_pause import ReplyOnPause, AppState, AlgoOptions, ReplyFnGenerator
from fastrtc.pause_detection import ModelOptions, PauseDetectionModel
from fastrtc.speech_to_text import get_stt_model, stt_for_chunks
from fastrtc.utils import audio_to_float32

logger = logging.getLogger(__name__)


@dataclass
class TurnState(AppState):
    """Extended state for turn detection"""
    conversation_history: List[Dict[str, str]] = None
    last_transcript: str = ""
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []


class TurnDetectionPause(ReplyOnPause):
    """
    Lean pause detector using semantic turn detection
    Combines VAD-based pause detection with LiveKit/TEN turn models
    """
    
    def __init__(
        self,
        fn: ReplyFnGenerator,
        turn_model: str = "livekit",  # "livekit" or "ten"
        turn_threshold: float = 0.5,
        stt_model: str = "moonshine/base",
        **kwargs
    ):
        """
        Args:
            fn: Reply function generator
            turn_model: "livekit" or "ten"
            turn_threshold: Confidence threshold for turn end (0.0-1.0)
            stt_model: STT model for transcription
            **kwargs: Other ReplyOnPause parameters
        """
        super().__init__(fn, **kwargs)
        
        self.turn_model_name = turn_model
        self.turn_threshold = turn_threshold
        self.stt_model = get_stt_model(stt_model)
        
        # Lazy initialization
        self._turn_detector = None
        self.state = TurnState()
    
    @property
    def turn_detector(self):
        """Lazy load turn detector"""
        if self._turn_detector is None:
            try:
                if self.turn_model_name == "livekit":
                    from .turn_detector import LiveKitTurnDetector
                    self._turn_detector = LiveKitTurnDetector()
                else:
                    from .turn_detector import TENTurnDetector
                    self._turn_detector = TENTurnDetector()
                logger.info(f"Turn detector loaded: {self.turn_model_name}")
            except Exception as e:
                logger.warning(f"Turn detector failed to load: {e}")
                self._turn_detector = None
        return self._turn_detector
    
    def determine_pause(self, audio: np.ndarray, sampling_rate: int, state: AppState) -> bool:
        """
        Enhanced pause detection with semantic turn analysis
        
        Strategy:
        1. Run standard VAD-based pause detection first
        2. If VAD detects pause, check semantic turn end
        3. Combine both signals for final decision
        """
        # Step 1: Standard VAD pause detection
        vad_pause = super().determine_pause(audio, sampling_rate, state)
        
        # If no VAD pause, no turn end
        if not vad_pause:
            return False
        
        # Step 2: Semantic turn detection
        if self.turn_detector is None:
            return vad_pause  # Fallback to VAD only
        
        return self._check_semantic_turn_end(state)
    
    def _check_semantic_turn_end(self, state: AppState) -> bool:
        """Check if conversation semantically indicates turn end"""
        try:
            # Get transcript from accumulated audio
            if state.stream is None or len(state.stream) == 0:
                return True  # Empty stream = end turn
            
            transcript = self._transcribe_audio(state.stream, state.sampling_rate)
            if not transcript.strip():
                return True  # No speech detected
            
            # Update conversation state
            self.state.last_transcript = transcript
            self._add_to_conversation("user", transcript)
            
            # Get turn end prediction
            result = self.turn_detector.should_end_turn(self.state.conversation_history)
            
            should_end = result.get("should_end", True)
            confidence = result.get("eou_probability", 0.5)
            
            logger.debug(f"Turn detection: {should_end} (confidence: {confidence:.2f})")
            
            return should_end and confidence > self.turn_threshold
            
        except Exception as e:
            logger.warning(f"Semantic turn detection failed: {e}")
            return True  # Fallback to ending turn
    
    def _transcribe_audio(self, audio: np.ndarray, sampling_rate: int) -> str:
        """Transcribe audio using STT model"""
        try:
            # Convert to float32 and resample to 16kHz for STT
            import librosa
            audio_f32 = audio_to_float32((sampling_rate, audio))
            audio_16k = librosa.resample(audio_f32, orig_sr=sampling_rate, target_sr=16000)
            
            # Get VAD chunks and transcribe
            _, chunks = self.model.vad((16000, audio_16k), self.model_options)
            text = stt_for_chunks(self.stt_model, (16000, audio_16k), chunks)
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Transcription failed: {e}")
            return ""
    
    def _add_to_conversation(self, role: str, content: str):
        """Add message to conversation history (keep last 6 messages)"""
        if content.strip():
            self.state.conversation_history.append({
                "role": role,
                "content": content
            })
            
            # Keep conversation history manageable
            if len(self.state.conversation_history) > 6:
                self.state.conversation_history = self.state.conversation_history[-6:]
    
    def reset(self):
        """Reset state for new conversation"""
        super().reset()
        self.state = TurnState()
    
    def copy(self):
        """Create a copy of this detector"""
        return TurnDetectionPause(
            self.fn,
            self.turn_model_name,
            self.turn_threshold,
            str(self.stt_model.__class__.__name__).lower().replace("stt", ""),
            startup_fn=self.startup_fn,
            algo_options=self.algo_options,
            model_options=self.model_options,
            can_interrupt=self.can_interrupt,
            expected_layout=self.expected_layout,
            output_sample_rate=self.output_sample_rate,
            input_sample_rate=self.input_sample_rate,
            model=self.model,
        )


# Convenience functions for quick setup
def create_livekit_pause_detector(fn: ReplyFnGenerator, threshold: float = 0.5, **kwargs):
    """Create LiveKit-based pause detector"""
    return TurnDetectionPause(
        fn=fn,
        turn_model="livekit",
        turn_threshold=threshold,
        **kwargs
    )


def create_ten_pause_detector(fn: ReplyFnGenerator, threshold: float = 0.5, **kwargs):
    """Create TEN-based pause detector"""
    return TurnDetectionPause(
        fn=fn,
        turn_model="ten", 
        turn_threshold=threshold,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    def example_reply_fn(audio_data):
        """Example reply function"""
        sample_rate, audio = audio_data
        print(f"Processing {len(audio)} samples at {sample_rate}Hz")
        yield audio  # Echo back
    
    # Create detector
    detector = create_livekit_pause_detector(
        fn=example_reply_fn,
        threshold=0.6,  # Higher threshold = more confident turn ends
        algo_options=AlgoOptions(
            audio_chunk_duration=0.5,  # Check every 500ms
            started_talking_threshold=0.2,
            speech_threshold=0.1
        )
    )
    
    print("Turn detection pause detector created successfully!")
    print(f"Using {detector.turn_model_name} model with threshold {detector.turn_threshold}")