# src/Agents/unified_call_center_agent/integration/fastrtc_integration.py
"""
FastRTC-specific integration for the unified call center agent
Handles the specific requirements and quirks of FastRTC audio streaming
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, Generator, List
from datetime import datetime

from .voice_integration import UnifiedVoiceHandler

logger = logging.getLogger(__name__)

class FastRTCUnifiedHandler(UnifiedVoiceHandler):
    """
    FastRTC-specific version of the unified voice handler.
    
    Handles:
    - FastRTC AdditionalOutputs constructor requirements
    - Audio stream safety checks
    - Proper error handling for WebRTC issues
    """
    
    def __init__(self, config: Dict[str, Any], workflow_factory=None):
        super().__init__(config, workflow_factory)
        self.audio_safety_checks = True
        logger.info("FastRTCUnifiedHandler initialized with safety checks")
    
    def process_audio_input(
        self,
        audio_input,
        state,  # Not used in unified system, kept for compatibility
        chatbot_history: List[Dict],
        thread_id: str
    ) -> Generator[Any, None, None]:
        """
        Process audio input with FastRTC-specific safety checks.
        
        Handles the WebRTC audio stream issues by adding safety checks
        and proper error handling.
        """
        
        if not self.current_workflow:
            yield self._create_fastrtc_error_response("No workflow available. Please load client data first.")
            return
        
        try:
            # Safety check for audio input
            if self.audio_safety_checks:
                if not self._validate_audio_input(audio_input):
                    logger.warning("Invalid audio input detected, using fallback")
                    audio_input = self._create_safe_audio_input()
            
            # Extract text from audio input (simplified for FastRTC)
            user_message = self._extract_message_from_audio(audio_input)
            
            if not user_message:
                yield self._create_fastrtc_error_response("No message could be extracted from audio input.")
                return
            
            logger.info(f"Processing audio input: {user_message[:50]}...")
            
            # Process the conversation turn
            response_data = self._process_conversation_turn(user_message, thread_id, chatbot_history)
            
            # Create FastRTC-compatible response
            yield self._create_fastrtc_response(response_data)
            
        except Exception as e:
            logger.error(f"Error in FastRTC audio processing: {e}")
            yield self._create_fastrtc_error_response(f"Audio processing error: {str(e)}")
    
    def _validate_audio_input(self, audio_input) -> bool:
        """
        Validate audio input to prevent FastRTC WebRTC errors.
        
        Checks for the NoneType issues that cause 'reshape' errors.
        """
        
        try:
            # Check if audio_input is None
            if audio_input is None:
                return False
            
            # Check if audio_input has expected attributes
            if hasattr(audio_input, 'stream'):
                if audio_input.stream is None:
                    logger.warning("Audio stream is None")
                    return False
            
            # Check if audio_input is a numpy array
            if isinstance(audio_input, np.ndarray):
                if audio_input.size == 0:
                    logger.warning("Empty audio array")
                    return False
            
            # Check for other common FastRTC issues
            if hasattr(audio_input, 'data'):
                if audio_input.data is None or len(audio_input.data) == 0:
                    logger.warning("Audio data is empty")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Audio validation error: {e}")
            return False
    
    def _create_safe_audio_input(self):
        """Create a safe audio input when validation fails"""
        
        # Create a minimal safe audio input structure
        class SafeAudioInput:
            def __init__(self):
                self.content = "Hello"  # Default message
                self.stream = np.array([0.0])  # Safe array
                self.data = [0.0]
        
        return SafeAudioInput()
    
    def _extract_message_from_audio(self, audio_input) -> str:
        """
        Extract text message from audio input.
        
        In a real implementation, this would use STT.
        For now, we handle different input types safely.
        """
        
        try:
            # Check various ways the message might be stored
            if hasattr(audio_input, 'content'):
                return str(audio_input.content)
            
            elif hasattr(audio_input, 'text'):
                return str(audio_input.text)
            
            elif hasattr(audio_input, 'message'):
                return str(audio_input.message)
            
            elif isinstance(audio_input, str):
                return audio_input
            
            elif isinstance(audio_input, dict) and 'content' in audio_input:
                return str(audio_input['content'])
            
            else:
                # Fallback - return a default message
                logger.warning("Could not extract message from audio input, using default")
                return "Hello"
        
        except Exception as e:
            logger.error(f"Error extracting message from audio: {e}")
            return "Hello"
    
    def _process_conversation_turn(
        self, 
        user_message: str, 
        thread_id: str, 
        chatbot_history: List[Dict]
    ) -> Dict[str, Any]:
        """Process conversation turn and return response data"""
        
        from ..agent.unified_workflow import create_conversation_config, get_conversation_state
        from langchain_core.messages import HumanMessage
        
        # Create conversation config
        config = create_conversation_config(thread_id)
        
        # Get current conversation state for context
        current_state = get_conversation_state(self.current_workflow, config)
        
        # Prepare input for workflow
        if current_state and current_state.get('messages'):
            # Continuing conversation
            workflow_input = {
                "messages": [HumanMessage(content=user_message)]
            }
        else:
            # Starting new conversation
            workflow_input = {}
        
        # Invoke workflow
        result = self.current_workflow.invoke(workflow_input, config)
        
        # Extract AI response
        ai_response = ""
        if result.get('messages'):
            last_message = result['messages'][-1]
            if hasattr(last_message, 'content'):
                ai_response = last_message.content
            else:
                ai_response = str(last_message)
        
        # Log conversation messages
        self._log_conversation_turn(user_message, ai_response, result)
        
        # Update chatbot history for frontend display
        updated_history = self._update_chatbot_history(
            chatbot_history, user_message, ai_response
        )
        
        return {
            "ai_response": ai_response,
            "updated_history": updated_history,
            "conversation_state": result
        }
    
    def _create_fastrtc_response(self, response_data: Dict[str, Any]) -> Any:
        """Create FastRTC-compatible response object"""
        
        try:
            from fastrtc import AdditionalOutputs
            
            # Use the correct constructor for FastRTC AdditionalOutputs
            # Based on the error, it expects positional arguments, not keyword arguments
            return AdditionalOutputs(
                response_data["updated_history"]  # First positional argument is chatbot history
            )
            
        except ImportError:
            logger.warning("FastRTC not available, using fallback response")
            return response_data
        
        except Exception as e:
            logger.error(f"Error creating FastRTC response: {e}")
            # Fallback response
            return {
                "chatbot_history": response_data.get("updated_history", []),
                "ai_response": response_data.get("ai_response", ""),
                "error": f"Response creation error: {str(e)}"
            }
    
    def _create_fastrtc_error_response(self, error_message: str) -> Any:
        """Create FastRTC-compatible error response"""
        
        error_history = [{
            "role": "assistant",
            "content": error_message,
            "timestamp": datetime.now().isoformat()
        }]
        
        try:
            from fastrtc import AdditionalOutputs
            
            # Use positional argument for error response
            return AdditionalOutputs(error_history)
            
        except ImportError:
            logger.warning("FastRTC not available, using fallback error response")
            return {
                "chatbot_history": error_history,
                "error": error_message
            }
        
        except Exception as e:
            logger.error(f"Error creating FastRTC error response: {e}")
            return {
                "chatbot_history": error_history,
                "error": error_message,
                "additional_error": str(e)
            }


def create_fastrtc_voice_handler(config: Dict[str, Any]) -> FastRTCUnifiedHandler:
    """
    Create FastRTC-specific unified voice handler.
    
    Args:
        config: Configuration dictionary from app_config
        
    Returns:
        FastRTCUnifiedHandler ready for use with FastRTC
    """
    
    def fastrtc_workflow_factory(client_data: Dict[str, Any]):
        """Create workflow optimized for FastRTC usage"""
        
        from langchain_ollama import ChatOllama
        from ..agent.unified_workflow import create_voice_compatible_workflow
        
        # Use smaller model for faster response times with FastRTC
        llm_config = config.get("llm", {})
        model = ChatOllama(
            model="qwen2.5:7b-instruct",  # Faster model for real-time audio
            temperature=0,
            num_ctx=16384  # Smaller context for speed
        )
        
        return create_voice_compatible_workflow(
            model=model,
            client_data=client_data,
            config=config,
            agent_name="Sarah"
        )
    
    handler = FastRTCUnifiedHandler(config, fastrtc_workflow_factory)
    logger.info("FastRTC unified voice handler created")
    
    return handler


# Compatibility wrapper for existing VoiceInteractionHandler interface
class FastRTCVoiceInteractionHandler:
    """
    FastRTC-specific compatibility wrapper that provides the same interface 
    as the existing VoiceInteractionHandler but handles FastRTC issues.
    """
    
    def __init__(self, config: Dict[str, Any], workflow_factory=None):
        # Create FastRTC-specific handler internally
        self.unified_handler = create_fastrtc_voice_handler(config)
        
        # Expose the same interface for compatibility
        self.config = config
        self.workflow = None
        self.message_streamer = None
        
        logger.info("FastRTCVoiceInteractionHandler (unified) initialized")
    
    def update_client_data(self, user_id: str, client_data: Dict[str, Any]):
        """Compatibility method"""
        self.unified_handler.update_client_data(user_id, client_data)
        # Set workflow property for compatibility
        self.workflow = self.unified_handler.current_workflow
    
    def process_audio_input(self, audio_input, state, chatbot_history: List[Dict], thread_id: str):
        """Compatibility method with FastRTC safety"""
        return self.unified_handler.process_audio_input(audio_input, state, chatbot_history, thread_id)
    
    def add_message_handler(self, handler):
        """Compatibility method"""
        self.unified_handler.add_message_handler(handler)
        
        # Create message_streamer compatibility object
        class MessageStreamer:
            def __init__(self, handlers):
                self.handlers = handlers
        
        self.message_streamer = MessageStreamer(self.unified_handler.message_handlers)


# Quick fix function for immediate integration
def get_fastrtc_compatible_handler(config: Dict[str, Any]):
    """
    Quick function to get a FastRTC-compatible handler.
    
    Use this as a drop-in replacement in your voice_chat_test_app.py
    """
    
    return FastRTCVoiceInteractionHandler(config)


# Testing function
def test_fastrtc_integration():
    """Test FastRTC integration with safety checks"""
    
    # Sample configuration
    test_config = {
        "llm": {
            "model_name": "qwen2.5:7b-instruct",
            "temperature": 0,
            "context_window": 16384
        },
        "configurable": {
            "use_memory": True
        }
    }
    
    # Sample client data
    test_client_data = {
        'user_id': '12345',
        'profile': {
            'user_id': '12345',
            'client_info': {
                'client_full_name': 'John Smith',
                'first_name': 'John',
                'title': 'Mr'
            }
        },
        'account_aging': {
            'xbalance': '299.00',
            'x0': '0.00',
            'x30': '299.00'
        }
    }
    
    # Create FastRTC handler
    voice_handler = create_fastrtc_voice_handler(test_config)
    
    # Update with client data
    voice_handler.update_client_data('12345', test_client_data)
    
    # Test with various audio input types
    test_inputs = [
        "Hello",  # String input
        None,     # None input (should be handled safely)
        {"content": "Yes, this is John"},  # Dict input
    ]
    
    thread_id = "test_fastrtc_001"
    chatbot_history = []
    
    print("=== Testing FastRTC Integration ===")
    
    for i, test_input in enumerate(test_inputs):
        print(f"\nTest {i+1}: {type(test_input).__name__} input")
        
        try:
            for response in voice_handler.process_audio_input(test_input, None, chatbot_history, thread_id):
                if hasattr(response, 'chatbot_history') and response.chatbot_history:
                    chatbot_history = response.chatbot_history
                    print(f"Success: {chatbot_history[-1]['content'][:50]}...")
                elif isinstance(response, dict) and 'chatbot_history' in response:
                    chatbot_history = response['chatbot_history']
                    print(f"Success: {chatbot_history[-1]['content'][:50]}...")
                else:
                    print(f"Response: {str(response)[:50]}...")
                break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nFastRTC integration test completed")
    return voice_handler

if __name__ == "__main__":
    test_fastrtc_integration()