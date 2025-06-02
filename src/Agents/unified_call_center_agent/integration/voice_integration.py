# src/Agents/unified_call_center_agent/integration/voice_integration.py
"""
Voice Integration Adapter - Seamlessly integrates unified agent with existing voice chat frontend
"""
import logging
import uuid
from typing import Dict, Any, Optional, Generator, List
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.graph import CompiledGraph

from ..agent.unified_workflow import (
    create_voice_compatible_workflow,
    create_conversation_config,
    get_conversation_state,
    get_conversation_history
)
from ..core.unified_agent_state import UnifiedAgentState

logger = logging.getLogger(__name__)

class UnifiedVoiceHandler:
    """
    Voice handler that integrates the unified agent with existing voice chat frontend.
    
    Provides the same interface as the current VoiceInteractionHandler but uses
    the new unified agent system internally for superior conversation quality.
    """
    
    def __init__(self, config: Dict[str, Any], workflow_factory=None):
        self.config = config
        self.workflow_factory = workflow_factory or self._default_workflow_factory
        self.current_workflow = None
        self.current_client_data = None
        self.message_handlers = []
        
        # Compatibility with existing voice handler interface
        self.model = self._create_model()
        
        logger.info("UnifiedVoiceHandler initialized")
    
    def _create_model(self) -> BaseChatModel:
        """Create language model from config"""
        from langchain_ollama import ChatOllama
        
        llm_config = self.config.get("llm", {})
        return ChatOllama(
            model=llm_config.get("model_name", "qwen2.5:14b-instruct"),
            temperature=llm_config.get("temperature", 0),
            num_ctx=llm_config.get("context_window", 32000)
        )
    
    def _default_workflow_factory(self, client_data: Dict[str, Any]) -> CompiledGraph:
        """Default workflow factory if none provided"""
        return create_voice_compatible_workflow(
            model=self.model,
            client_data=client_data,
            config=self.config,
            agent_name="Sarah"
        )
    
    def update_client_data(self, user_id: str, client_data: Dict[str, Any]):
        """
        Update client data and create new workflow.
        
        Args:
            user_id: Client identifier
            client_data: Client information and account data
        """
        
        logger.info(f"Updating client data for user: {user_id}")
        
        self.current_client_data = client_data
        
        # Create new workflow for this client
        self.current_workflow = self.workflow_factory(client_data)
        
        client_name = client_data.get('profile', {}).get('client_info', {}).get('client_full_name', 'Unknown')
        logger.info(f"Workflow ready for client: {client_name}")
    
    def process_audio_input(
        self,
        audio_input,
        state,  # Not used in unified system, kept for compatibility
        chatbot_history: List[Dict],
        thread_id: str
    ) -> Generator[Any, None, None]:
        """
        Process audio input and generate response.
        
        Compatible with existing voice chat frontend interface.
        
        Args:
            audio_input: Audio input from user
            state: Legacy state (ignored)
            chatbot_history: Chat history for display
            thread_id: Conversation thread identifier
            
        Yields:
            Response objects compatible with existing frontend
        """
        
        if not self.current_workflow:
            yield self._create_error_response("No workflow available. Please load client data first.")
            return
        
        try:
            # Create conversation config
            config = create_conversation_config(thread_id)
            
            # Simulate audio processing (in real implementation, would use STT)
            # For now, assume audio_input contains the transcribed text
            if hasattr(audio_input, 'content'):
                user_message = audio_input.content
            elif isinstance(audio_input, str):
                user_message = audio_input
            else:
                user_message = "Hello"  # Default for testing
            
            logger.info(f"Processing audio input: {user_message[:50]}...")
            
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
            
            # Create response object compatible with existing frontend
            response_obj = self._create_voice_response(
                ai_response, updated_history, result
            )
            
            yield response_obj
            
        except Exception as e:
            logger.error(f"Error processing audio input: {e}")
            yield self._create_error_response(f"Processing error: {str(e)}")
    
    def _log_conversation_turn(self, user_message: str, ai_response: str, result: Dict[str, Any]):
        """Log conversation turn for handlers"""
        
        # Call registered message handlers
        for handler in self.message_handlers:
            try:
                handler("human", user_message)
                handler("assistant", ai_response)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
        
        # Log conversation metadata
        metadata = {
            "turn_count": result.get("turn_count", 0),
            "current_objective": result.get("current_objective", "unknown"),
            "client_mood": result.get("client_mood", "neutral"),
            "last_action": result.get("last_action", "unknown"),
            "last_intent": result.get("last_intent", "none")
        }
        
        logger.info(f"Conversation turn logged - {metadata}")
    
    def _update_chatbot_history(
        self, 
        current_history: List[Dict], 
        user_message: str, 
        ai_response: str
    ) -> List[Dict]:
        """Update chatbot history for frontend display"""
        
        updated_history = current_history.copy()
        
        # Add user message
        updated_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add AI response
        updated_history.append({
            "role": "assistant", 
            "content": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep history manageable (last 20 messages)
        if len(updated_history) > 20:
            updated_history = updated_history[-20:]
        
        return updated_history
    
    def _create_voice_response(
        self, 
        ai_response: str, 
        updated_history: List[Dict], 
        result: Dict[str, Any]
    ) -> Any:
        """Create response object compatible with existing voice frontend"""
        
        # Import the response class from existing system
        try:
            from fastrtc import AdditionalOutputs
            
            # FastRTC AdditionalOutputs expects specific parameters
            # Check the actual constructor signature
            return AdditionalOutputs(
                updated_history,  # First positional argument is the chatbot history
                additional_outputs={
                    "ai_response": ai_response,
                    "conversation_state": {
                        "turn_count": result.get("turn_count", 0),
                        "current_objective": result.get("current_objective", "unknown"),
                        "client_mood": result.get("client_mood", "neutral"),
                        "verification_status": {
                            "name": result.get("name_verification", "pending"),
                            "details": result.get("details_verification", "pending")
                        },
                        "call_ended": result.get("call_ended", False),
                        "objectives_completed": len(result.get("completed_objectives", []))
                    }
                }
            )
            
        except (ImportError, TypeError) as e:
            logger.warning(f"FastRTC AdditionalOutputs error: {e}, using fallback")
            # Fallback if FastRTC not available or constructor changed
            return {
                "chatbot_history": updated_history,
                "ai_response": ai_response,
                "conversation_state": result
            }
    
    def _create_error_response(self, error_message: str) -> Any:
        """Create error response compatible with frontend"""
        
        error_history = [{
            "role": "assistant",
            "content": error_message,
            "timestamp": datetime.now().isoformat()
        }]
        
        try:
            from fastrtc import AdditionalOutputs
            
            # Use positional argument for chatbot history
            return AdditionalOutputs(error_history)
            
        except (ImportError, TypeError) as e:
            logger.warning(f"FastRTC AdditionalOutputs error: {e}, using fallback")
            return {
                "chatbot_history": error_history,
                "error": error_message
            }
    
    def add_message_handler(self, handler):
        """Add message handler for logging/monitoring"""
        self.message_handlers.append(handler)
        logger.info("Message handler added")
    
    def get_conversation_status(self, thread_id: str) -> Dict[str, Any]:
        """Get current conversation status"""
        
        if not self.current_workflow:
            return {"error": "No workflow available"}
        
        try:
            config = create_conversation_config(thread_id)
            state = get_conversation_state(self.current_workflow, config)
            
            if state:
                return {
                    "active": True,
                    "turn_count": state.get("turn_count", 0),
                    "current_objective": state.get("current_objective", "unknown"),
                    "client_mood": state.get("client_mood", "neutral"),
                    "verification_completed": state.get("name_verification") == "VERIFIED" and state.get("details_verification") == "VERIFIED",
                    "payment_secured": state.get("payment_secured", False),
                    "call_ended": state.get("call_ended", False),
                    "objectives_completed": len(state.get("completed_objectives", []))
                }
            else:
                return {"active": False, "message": "No conversation state found"}
                
        except Exception as e:
            logger.error(f"Error getting conversation status: {e}")
            return {"error": str(e)}
    
    def clear_conversation(self, thread_id: str) -> bool:
        """Clear conversation history"""
        
        if not self.current_workflow:
            return False
        
        try:
            from ..agent.unified_workflow import clear_conversation_history
            
            config = create_conversation_config(thread_id)
            return clear_conversation_history(self.current_workflow, config)
            
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            return False

# Factory function for easy integration
def create_unified_voice_handler(config: Dict[str, Any]) -> UnifiedVoiceHandler:
    """
    Create unified voice handler for integration with existing frontend.
    
    Args:
        config: Configuration dictionary from app_config
        
    Returns:
        UnifiedVoiceHandler ready for use
    """
    
    def workflow_factory(client_data: Dict[str, Any]) -> CompiledGraph:
        """Create workflow with appropriate model selection"""
        
        from langchain_ollama import ChatOllama
        
        # Select model based on conversation complexity
        llm_config = config.get("llm", {})
        model = ChatOllama(
            model=llm_config.get("model_name", "qwen2.5:14b-instruct"),
            temperature=llm_config.get("temperature", 0),
            num_ctx=llm_config.get("context_window", 32000)
        )
        
        return create_voice_compatible_workflow(
            model=model,
            client_data=client_data,
            config=config,
            agent_name="Sarah"
        )
    
    handler = UnifiedVoiceHandler(config, workflow_factory)
    logger.info("Unified voice handler created and ready for integration")
    
    return handler

# Compatibility layer for existing VoiceInteractionHandler interface
class VoiceInteractionHandler:
    """
    Compatibility wrapper that provides the same interface as the existing
    VoiceInteractionHandler but uses the unified agent internally.
    
    This allows seamless integration with existing voice chat frontend
    without requiring any changes to the frontend code.
    """
    
    def __init__(self, config: Dict[str, Any], workflow_factory=None):
        # Create unified handler internally
        self.unified_handler = UnifiedVoiceHandler(config, workflow_factory)
        
        # Expose the same interface
        self.config = config
        self.workflow = None  # Compatibility property
        
        logger.info("VoiceInteractionHandler (unified) initialized")
    
    def update_client_data(self, user_id: str, client_data: Dict[str, Any]):
        """Compatibility method"""
        self.unified_handler.update_client_data(user_id, client_data)
        # Set workflow property for compatibility
        self.workflow = self.unified_handler.current_workflow
    
    def process_audio_input(self, audio_input, state, chatbot_history: List[Dict], thread_id: str):
        """Compatibility method"""
        return self.unified_handler.process_audio_input(audio_input, state, chatbot_history, thread_id)
    
    def add_message_handler(self, handler):
        """Compatibility method"""
        self.unified_handler.add_message_handler(handler)
    
    # Additional compatibility properties/methods as needed
    @property
    def message_streamer(self):
        """Compatibility property"""
        class MessageStreamer:
            def __init__(self, handlers):
                self.handlers = handlers
        
        return MessageStreamer(self.unified_handler.message_handlers)

# Usage example for testing integration
def test_voice_integration():
    """Test voice integration with sample data"""
    
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
    
    # Create voice handler
    voice_handler = create_unified_voice_handler(test_config)
    
    # Update with client data
    voice_handler.update_client_data('12345', test_client_data)
    
    # Test conversation
    thread_id = "test_integration_001"
    chatbot_history = []
    
    print("=== Testing Voice Integration ===")
    
    # Process first message (should trigger greeting)
    for response in voice_handler.process_audio_input("Hello", None, chatbot_history, thread_id):
        if hasattr(response, 'chatbot_history'):
            chatbot_history = response.chatbot_history
            print(f"Agent: {chatbot_history[-1]['content']}")
        break
    
    # Process client confirmation
    for response in voice_handler.process_audio_input("Yes, this is John", None, chatbot_history, thread_id):
        if hasattr(response, 'chatbot_history'):
            chatbot_history = response.chatbot_history
            print(f"Agent: {chatbot_history[-1]['content']}")
        break
    
    # Check conversation status
    status = voice_handler.get_conversation_status(thread_id)
    print(f"Conversation status: {status}")
    
    return voice_handler

if __name__ == "__main__":
    test_voice_integration()