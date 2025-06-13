# src/VoiceHandler/turn_detection.py
"""
LiveKit Turn Detection Integration for Voice Agent Pipeline
Integrates LiveKit's semantic turn detection with existing FastRTC pipeline
"""

import asyncio
import logging
import numpy as np
import time
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from livekit.plugins.turn_detector import MultilingualModel
    from livekit.plugins.turn_detector.base import TurnDetector, TurnDetectorEvent
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    # Fallback stubs for when LiveKit is not available
    class TurnDetector:
        pass
    
    class TurnDetectorEvent:
        pass
    
    class MultilingualModel:
        pass

logger = logging.getLogger(__name__)


class TurnState(Enum):
    """Turn detection states"""
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    TURN_DETECTED = "turn_detected"
    SILENCE = "silence"


@dataclass
class TurnDetectionConfig:
    """Configuration for turn detection"""
    # Model settings
    model_name: str = "multilingual"  # or "eou" for End-of-Utterance
    confidence_threshold: float = 0.7
    
    # Timing settings
    min_speech_duration: float = 0.5  # Minimum speech duration before considering turn
    max_silence_duration: float = 2.0  # Maximum silence before forcing turn
    processing_timeout: float = 5.0   # Timeout for LLM processing
    
    # Context window
    context_window_turns: int = 4  # Number of previous turns to consider
    
    # Fallback VAD settings
    fallback_to_vad: bool = True
    vad_silence_threshold: float = 1.0  # Fallback VAD silence threshold


class LiveKitTurnDetector:
    """
    LiveKit turn detection integration for existing voice pipeline
    Provides semantic turn detection with VAD fallback
    """
    
    def __init__(self, config: TurnDetectionConfig = None):
        self.config = config or TurnDetectionConfig()
        self.state = TurnState.LISTENING
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_speech_start: Optional[float] = None
        self.last_speech_end: Optional[float] = None
        self.is_processing_llm = False
        
        # Callbacks
        self.on_turn_detected: Optional[Callable] = None
        self.on_state_changed: Optional[Callable] = None
        
        # Initialize turn detector
        self._init_turn_detector()
        
        logger.info(f"LiveKit Turn Detection initialized with model: {self.config.model_name}")
    
    def _init_turn_detector(self):
        """Initialize the LiveKit turn detector"""
        if not LIVEKIT_AVAILABLE:
            logger.warning("LiveKit not available, using fallback VAD-only detection")
            self.turn_detector = None
            return
        
        try:
            if self.config.model_name == "multilingual":
                self.turn_detector = MultilingualModel()
            else:
                # Can be extended for other models
                self.turn_detector = MultilingualModel()
                
            logger.info("LiveKit turn detector loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LiveKit turn detector: {e}")
            self.turn_detector = None
    
    def add_conversation_turn(self, role: str, content: str):
        """Add a conversation turn to the context window"""
        turn = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        self.conversation_history.append(turn)
        
        # Keep only the last N turns for context
        if len(self.conversation_history) > self.config.context_window_turns:
            self.conversation_history = self.conversation_history[-self.config.context_window_turns:]
    
    def set_processing_state(self, is_processing: bool):
        """Update LLM processing state"""
        self.is_processing_llm = is_processing
        if is_processing:
            self._update_state(TurnState.PROCESSING)
        else:
            self._update_state(TurnState.LISTENING)
    
    def _update_state(self, new_state: TurnState):
        """Update turn detection state and notify callbacks"""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            
            logger.debug(f"Turn state changed: {old_state.value} -> {new_state.value}")
            
            if self.on_state_changed:
                self.on_state_changed(old_state, new_state)
    
    def _get_conversation_context(self) -> str:
        """Get formatted conversation context for semantic analysis"""
        if not self.conversation_history:
            return ""
        
        context_lines = []
        for turn in self.conversation_history[-self.config.context_window_turns:]:
            role_label = "Human" if turn["role"] == "user" else "Assistant"
            context_lines.append(f"{role_label}: {turn['content']}")
        
        return "\n".join(context_lines)
    
    async def detect_turn(self, 
                         audio_chunk: Tuple[int, np.ndarray], 
                         transcript: str = "", 
                         is_speech_active: bool = False) -> bool:
        """
        Main turn detection logic combining semantic analysis with VAD
        
        Args:
            audio_chunk: Audio data (sample_rate, audio_array)
            transcript: Current transcript text
            is_speech_active: Whether speech is currently detected by VAD
            
        Returns:
            True if turn should be taken (end of user utterance detected)
        """
        current_time = time.time()
        
        # Track speech timing
        if is_speech_active:
            if self.current_speech_start is None:
                self.current_speech_start = current_time
            self.last_speech_end = current_time
        else:
            if self.current_speech_start is not None:
                # Speech just ended
                speech_duration = current_time - self.current_speech_start
                if speech_duration < self.config.min_speech_duration:
                    # Too short, ignore
                    self.current_speech_start = None
                    return False
                
                self.current_speech_start = None
        
        # Don't interrupt if we're processing
        if self.is_processing_llm:
            return False
        
        # Semantic turn detection if available
        if self.turn_detector and transcript:
            try:
                semantic_result = await self._semantic_turn_detection(transcript)
                if semantic_result:
                    self._update_state(TurnState.TURN_DETECTED)
                    return True
            except Exception as e:
                logger.error(f"Semantic turn detection error: {e}")
                # Fall back to VAD-based detection
        
        # Fallback VAD-based turn detection
        if self.config.fallback_to_vad:
            vad_result = self._vad_turn_detection(is_speech_active, current_time)
            if vad_result:
                self._update_state(TurnState.TURN_DETECTED)
                return True
        
        return False
    
    async def _semantic_turn_detection(self, transcript: str) -> bool:
        """Semantic turn detection using LiveKit model"""
        if not self.turn_detector or not transcript.strip():
            return False
        
        try:
            # Get conversation context
            context = self._get_conversation_context()
            
            # Prepare input for turn detector
            # Note: Actual LiveKit API may differ - adjust based on documentation
            detection_input = {
                "transcript": transcript,
                "context": context,
                "conversation_history": self.conversation_history
            }
            
            # Run semantic analysis
            # This is a placeholder - adjust based on actual LiveKit API
            result = await self.turn_detector.detect(detection_input)
            
            # Check confidence threshold
            if hasattr(result, 'confidence'):
                return result.confidence >= self.config.confidence_threshold
            elif hasattr(result, 'is_turn_complete'):
                return result.is_turn_complete
            else:
                # Default interpretation
                return bool(result)
                
        except Exception as e:
            logger.error(f"Semantic turn detection failed: {e}")
            return False
    
    def _vad_turn_detection(self, is_speech_active: bool, current_time: float) -> bool:
        """Fallback VAD-based turn detection"""
        if is_speech_active:
            self._update_state(TurnState.LISTENING)
            return False
        
        # Check silence duration
        if self.last_speech_end:
            silence_duration = current_time - self.last_speech_end
            if silence_duration >= self.config.vad_silence_threshold:
                return True
        
        return False
    
    def reset(self):
        """Reset turn detection state"""
        self.state = TurnState.LISTENING
        self.current_speech_start = None
        self.last_speech_end = None
        self.is_processing_llm = False
        
        logger.debug("Turn detection state reset")


class EnhancedVoiceInteractionHandler:
    """
    Enhanced VoiceInteractionHandler with LiveKit turn detection integration
    """
    
    def __init__(self, config: Dict[str, Any], workflow_factory: Optional[Callable] = None):
        # Initialize existing components
        from src.VoiceHandler.voice_handler import VoiceInteractionHandler
        self.base_handler = VoiceInteractionHandler(config, workflow_factory)
        
        # Initialize turn detection
        turn_config = TurnDetectionConfig(
            confidence_threshold=config.get('turn_detection', {}).get('confidence_threshold', 0.7),
            min_speech_duration=config.get('turn_detection', {}).get('min_speech_duration', 0.5),
            max_silence_duration=config.get('turn_detection', {}).get('max_silence_duration', 2.0),
            fallback_to_vad=config.get('turn_detection', {}).get('fallback_to_vad', True)
        )
        
        self.turn_detector = LiveKitTurnDetector(turn_config)
        
        # Set up callbacks
        self.turn_detector.on_turn_detected = self._on_turn_detected
        self.turn_detector.on_state_changed = self._on_state_changed
        
        # Turn detection state
        self.current_transcript = ""
        self.pending_audio_chunks = []
        
        logger.info("Enhanced VoiceInteractionHandler with turn detection initialized")
    
    def _on_turn_detected(self):
        """Callback when turn is detected"""
        logger.debug("Turn detected - processing user input")
        # Could trigger immediate processing or other actions
    
    def _on_state_changed(self, old_state: TurnState, new_state: TurnState):
        """Callback when turn detection state changes"""
        logger.debug(f"Turn detection state: {old_state.value} -> {new_state.value}")
        
        # Could update UI indicators or other components
        if hasattr(self.base_handler, 'message_streamer'):
            self.base_handler.message_streamer.stream_message(
                "system", f"Turn state: {new_state.value}"
            )
    
    async def process_audio_with_turn_detection(self, 
                                               audio_input: Tuple[int, np.ndarray],
                                               workflow = None,
                                               gradio_chatbot = None,
                                               thread_id: str = None) -> Dict[str, Any]:
        """
        Process audio with enhanced turn detection
        
        Args:
            audio_input: Audio data (sample_rate, audio_array)
            workflow: Workflow instance
            gradio_chatbot: Gradio chatbot history
            thread_id: Thread ID for conversation
            
        Returns:
            Processing results with turn detection info
        """
        try:
            # Validate audio input
            if not self.base_handler._validate_audio_input(audio_input):
                return {"error": "Invalid audio input"}
            
            # Get STT transcription
            if (self.base_handler.stt_model and 
                self.base_handler.health_monitor.is_healthy('stt')):
                
                try:
                    stt_result = self.base_handler.stt_model.transcribe(audio_input)
                    transcript = (stt_result.get("text", "") if isinstance(stt_result, dict) 
                                else str(stt_result)).strip()
                    self.current_transcript = transcript
                except Exception as e:
                    logger.error(f"STT failed: {e}")
                    transcript = ""
            else:
                transcript = ""
            
            # Simple VAD detection (you might want to use your existing VAD)
            sample_rate, audio_array = audio_input
            is_speech_active = self._detect_speech_activity(audio_array)
            
            # Run turn detection
            should_process = await self.turn_detector.detect_turn(
                audio_input, transcript, is_speech_active
            )
            
            if should_process and transcript:
                # Set processing state
                self.turn_detector.set_processing_state(True)
                
                try:
                    # Process through existing pipeline
                    result = self.base_handler.process_message(transcript, workflow)
                    
                    # Add to conversation history
                    self.turn_detector.add_conversation_turn("user", transcript)
                    
                    # Extract AI response for context
                    ai_response = self.base_handler._extract_ai_response(result)
                    if ai_response:
                        self.turn_detector.add_conversation_turn("assistant", ai_response)
                    
                    return {
                        "transcript": transcript,
                        "workflow_result": result,
                        "turn_detected": True,
                        "turn_state": self.turn_detector.state.value
                    }
                    
                finally:
                    # Reset processing state
                    self.turn_detector.set_processing_state(False)
            
            else:
                return {
                    "transcript": transcript,
                    "turn_detected": False,
                    "turn_state": self.turn_detector.state.value,
                    "is_speech_active": is_speech_active
                }
                
        except Exception as e:
            logger.error(f"Audio processing with turn detection failed: {e}")
            return {"error": str(e)}
    
    def _detect_speech_activity(self, audio_array: np.ndarray) -> bool:
        """Simple speech activity detection based on energy"""
        if len(audio_array) == 0:
            return False
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_array.astype(float) ** 2))
        
        # Simple threshold-based detection
        threshold = 0.01  # Adjust based on your audio levels
        return rms > threshold
    
    # Delegate other methods to base handler
    def __getattr__(self, name):
        """Delegate unknown attributes to base handler"""
        return getattr(self.base_handler, name)


# Updated app_config.py additions
TURN_DETECTION_CONFIG = {
    "turn_detection": {
        "enabled": True,
        "model_name": "multilingual",
        "confidence_threshold": 0.7,
        "min_speech_duration": 0.5,
        "max_silence_duration": 2.0,
        "fallback_to_vad": True,
        "context_window_turns": 4
    }
}


# Example usage in your main application
def create_enhanced_voice_handler(config: Dict[str, Any], workflow_factory):
    """Create voice handler with turn detection"""
    
    # Merge turn detection config
    enhanced_config = {**config, **TURN_DETECTION_CONFIG}
    
    # Create enhanced handler
    return EnhancedVoiceInteractionHandler(enhanced_config, workflow_factory)


# Example integration with your existing FastRTC setup
async def enhanced_audio_callback(audio_input, enhanced_handler, workflow, chatbot, thread_id):
    """Enhanced audio callback with turn detection"""
    
    result = await enhanced_handler.process_audio_with_turn_detection(
        audio_input, workflow, chatbot, thread_id
    )
    
    if result.get("turn_detected"):
        logger.info(f"Turn detected - processing: {result.get('transcript', '')}")
        
        # Continue with your existing processing logic
        # The workflow_result contains the processed response
        return result.get("workflow_result")
    else:
        logger.debug(f"No turn detected - state: {result.get('turn_state')}")
        return None