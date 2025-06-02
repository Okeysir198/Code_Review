# src/Agents/unified_call_center_agent/integration/voice_integration.py
"""
Enhanced Voice integration with better error handling and debugging
"""

import logging
import uuid
import json
from typing import Dict, Any, Optional, Tuple, List, Generator, Union, Set, Callable
import numpy as np
from fastrtc import AdditionalOutputs
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

logger = logging.getLogger(__name__)

# Try to import STT/TTS models - these need to exist in your project
try:
    from src.STT import create_stt_model, BaseSTTModel
    STT_AVAILABLE = True
except ImportError:
    logging.warning("STT module not found - STT functionality disabled")
    BaseSTTModel = object
    STT_AVAILABLE = False
    def create_stt_model(config): return None

try:
    from src.TTS import create_tts_model, BaseTTSModel  
    TTS_AVAILABLE = True
except ImportError:
    logging.warning("TTS module not found - TTS functionality disabled")
    BaseTTSModel = object
    TTS_AVAILABLE = False
    def create_tts_model(config): return None


class MessageStreamer:
    """Handles real-time message streaming to external handlers."""
    
    def __init__(self):
        self.handlers = []
    
    def add_handler(self, handler: Callable[[str, str], None]):
        """Add a message handler function that takes (role, content)."""
        self.handlers.append(handler)
    
    def stream_message(self, role: str, content: str):
        """Stream message to all registered handlers."""
        for handler in self.handlers:
            try:
                handler(role, content)
            except Exception as e:
                logger.error(f"Message handler error: {e}")


class UnifiedVoiceHandler:
    """Voice handler using the unified call center agent with enhanced debugging."""
    
    def __init__(self, config: Dict[str, Any], workflow_factory: Optional[Callable] = None):
        self.config = self._setup_config(config or {})
        self.workflow_factory = workflow_factory
        self.message_streamer = MessageStreamer()
        self._setup_logging()
        
        # Client data caching with debug info
        self.cached_client_data = None
        self.cached_user_id = None
        self.cached_workflow = None
        self.workflow_creation_error = None  # Track workflow creation errors
        
        # Debug logging
        logger.info(f"UnifiedVoiceHandler initialized with workflow_factory: {workflow_factory is not None}")
        
        # Initialize models
        self.stt_model = self._init_stt() if self.config['configurable'].get('enable_stt_model') else None
        self.tts_model = self._init_tts() if self.config['configurable'].get('enable_tts_model') else None

    def add_message_handler(self, handler: Callable[[str, str], None]):
        """Add external message handler for real-time streaming."""
        self.message_streamer.add_handler(handler)

    def update_client_data(self, user_id: str, client_data: Dict[str, Any]):
        """Update cached client data and workflow with enhanced debugging."""
        try:
            logger.info(f"🔄 update_client_data called for user_id: {user_id}")
            logger.info(f"📊 Client data keys: {list(client_data.keys()) if client_data else 'None'}")
            logger.info(f"🏭 Workflow factory available: {self.workflow_factory is not None}")
            
            if not user_id or not user_id.strip():
                logger.error("❌ No user_id provided")
                return
                
            if not client_data:
                logger.error("❌ No client_data provided")
                return
                
            if not self.workflow_factory:
                logger.error("❌ No workflow_factory available")
                self.workflow_creation_error = "No workflow factory provided"
                return
            
            # Update cached data
            self.cached_user_id = user_id
            self.cached_client_data = client_data
            self.workflow_creation_error = None
            
            # Create new workflow
            logger.info(f"🔧 Creating workflow for user_id: {user_id}")
            try:
                self.cached_workflow = self.workflow_factory(client_data)
                if self.cached_workflow:
                    logger.info(f"✅ Workflow created successfully for user_id: {user_id}")
                else:
                    logger.error(f"❌ Workflow factory returned None for user_id: {user_id}")
                    self.workflow_creation_error = "Workflow factory returned None"
            except Exception as e:
                logger.error(f"❌ Workflow creation failed for user_id {user_id}: {e}")
                self.workflow_creation_error = str(e)
                self.cached_workflow = None
                import traceback
                traceback.print_exc()
                    
        except Exception as e:
            logger.error(f"❌ Error in update_client_data for user_id {user_id}: {e}")
            self.workflow_creation_error = str(e)
            self.cached_workflow = None
            import traceback
            traceback.print_exc()

    def get_current_workflow(self) -> Optional[CompiledGraph]:
        """Get the current cached workflow with debugging."""
        logger.debug(f"🔍 get_current_workflow called")
        logger.debug(f"📋 Cached workflow available: {self.cached_workflow is not None}")
        logger.debug(f"👤 Cached user_id: {self.cached_user_id}")
        
        if self.workflow_creation_error:
            logger.error(f"❌ Previous workflow creation error: {self.workflow_creation_error}")
        
        return self.cached_workflow

    def _setup_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Setup configuration with defaults."""
        config.setdefault('stt', {})
        config.setdefault('tts', {})
        config.setdefault('logging', {'level': 'info', 'console_output': True})  # Changed to info for debugging
        config.setdefault('configurable', {
            'thread_id': str(uuid.uuid4()),
            'enable_stt_model': STT_AVAILABLE,
            'enable_tts_model': TTS_AVAILABLE
        })
        return config

    def _setup_logging(self):
        """Configure logging."""
        level_map = {
            'none': logging.CRITICAL + 1,
            'error': logging.ERROR,
            'warning': logging.WARNING,
            'info': logging.INFO,
            'debug': logging.DEBUG
        }
        
        level = level_map.get(self.config['logging'].get('level', 'info').lower(), logging.INFO)
        logger.setLevel(level)

    def _init_stt(self) -> Optional[BaseSTTModel]:
        """Initialize STT model."""
        try:
            if STT_AVAILABLE and create_stt_model:
                return create_stt_model(self.config['stt'])
        except Exception as e:
            logger.error(f"STT init failed: {e}")
        return None

    def _init_tts(self) -> Optional[BaseTTSModel]:
        """Initialize TTS model."""
        try:
            if TTS_AVAILABLE and create_tts_model:
                return create_tts_model(self.config['tts'])
        except Exception as e:
            logger.error(f"TTS init failed: {e}")
        return None

    def _extract_ai_response(self, workflow_result: Dict[str, Any]) -> str:
        """Extract AI response from workflow result."""
        if "error" in workflow_result:
            return workflow_result["error"]
        
        messages = workflow_result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
                return msg.content
        
        return "No AI response found."

    def process_message(self, user_message: str, workflow: CompiledGraph = None) -> Dict[str, Any]:
        """Process message through workflow with enhanced debugging."""
        logger.info(f"💬 process_message called with: '{user_message}'")
        
        # Use cached workflow if no workflow provided
        if workflow is None:
            workflow = self.get_current_workflow()
            logger.info(f"🔍 Using cached workflow: {workflow is not None}")
        
        if not workflow:
            error_msg = "No workflow available. Please select a client first."
            if self.workflow_creation_error:
                error_msg += f" Error: {self.workflow_creation_error}"
            
            logger.error(f"❌ {error_msg}")
            logger.error(f"🔧 Debug info:")
            logger.error(f"   - Cached user_id: {self.cached_user_id}")
            logger.error(f"   - Client data available: {self.cached_client_data is not None}")
            logger.error(f"   - Workflow factory available: {self.workflow_factory is not None}")
            logger.error(f"   - Workflow creation error: {self.workflow_creation_error}")
            
            return {"messages": [], "error": error_msg}
        
        try:
            workflow_input = {"messages": [HumanMessage(content=user_message)]}
            config = {"configurable": self.config.get('configurable', {})}
            
            logger.info(f"🚀 Invoking workflow with config: {config}")
            
            # Stream user message
            self.message_streamer.stream_message("user", user_message)
            
            # Run workflow
            result = workflow.invoke(workflow_input, config=config)
            logger.info(f"✅ Workflow completed successfully")
            
            # Stream AI response
            ai_response = self._extract_ai_response(result)
            if ai_response:
                self.message_streamer.stream_message("ai", ai_response)
            
            return result
            
        except Exception as e:
            error_msg = f"Workflow execution error: {str(e)}"
            logger.error(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            
            self.message_streamer.stream_message("error", error_msg)
            return {"messages": [], "error": error_msg}

    def process_audio_input(self, audio_input: Tuple[int, np.ndarray], workflow: CompiledGraph = None,
                           gradio_chatbot: Optional[List[Dict[str, str]]] = None,
                           thread_id: Optional[Union[str, int]] = None) -> Generator:
        """Process audio input with enhanced debugging."""
        try:
            chatbot = gradio_chatbot or []
            sample_rate = audio_input[0]
            
            # Check if audio is valid
            if (not audio_input[1].size or not self.stt_model or 
                not self.config['configurable'].get('enable_stt_model')):
                logger.warning("⚠️ Audio processing skipped - no audio data or STT disabled")
                yield (sample_rate, np.array([], dtype=np.int16))
                yield AdditionalOutputs(self._format_chatbot(chatbot))
                return
            
            # Transcribe audio
            stt_result = self.stt_model.transcribe(audio_input)
            stt_text = (stt_result.get("text", "") if isinstance(stt_result, dict) 
                       else str(stt_result)).strip()
            
            if not stt_text:
                logger.warning("⚠️ No text transcribed from audio")
                yield (sample_rate, np.array([], dtype=np.int16))
                yield AdditionalOutputs(self._format_chatbot(chatbot))
                return
            
            # Process conversation
            logger.info(f"🎤 Voice Input: {stt_text}")
            chatbot.append({"role": "user", "content": stt_text})
            
            # Set thread ID and get response with cached workflow
            self.config['configurable']["thread_id"] = str(thread_id or uuid.uuid4())
            workflow_result = self.process_message(stt_text, workflow)
            response = self._extract_ai_response(workflow_result)
            
            logger.info(f"🤖 AI Response: {response}")
            chatbot.append({"role": "assistant", "content": response})
            
            # Update UI with final chatbot state
            yield AdditionalOutputs(self._format_chatbot(chatbot))
            
            # Generate TTS
            if self.tts_model and self.config['configurable'].get('enable_tts_model') and response.strip():
                for chunk in self.tts_model.stream_text_to_speech(response):
                    yield self._normalize_audio(chunk)
            else:
                yield (sample_rate, np.array([0], dtype=np.int16))
            
        except Exception as e:
            logger.error(f"❌ Audio processing error: {e}")
            import traceback
            traceback.print_exc()
            
            chatbot.append({"role": "assistant", "content": "Sorry, an error occurred."})
            yield (sample_rate, np.array([0], dtype=np.int16))
            yield AdditionalOutputs(self._format_chatbot(chatbot))

    def _format_chatbot(self, chatbot: List[Dict]) -> List[Dict]:
        """Format chatbot messages."""
        return [
            {"role": str(msg.get("role", "")), "content": str(msg.get("content", ""))}
            for msg in chatbot
            if isinstance(msg, dict) and msg.get('role') and msg.get('content')
        ]

    def _normalize_audio(self, audio_chunk: tuple) -> tuple:
        """Normalize audio format to int16."""
        if isinstance(audio_chunk, tuple) and len(audio_chunk) == 2:
            sample_rate, audio_array = audio_chunk
            if hasattr(audio_array, 'dtype') and audio_array.dtype == np.float32:
                audio_array = (audio_array * 32767).astype(np.int16)
                return (sample_rate, audio_array)
        return audio_chunk


# Compatibility wrapper for existing frontend
class VoiceInteractionHandler(UnifiedVoiceHandler):
    """
    Compatibility wrapper that provides the same interface as the original
    VoiceInteractionHandler but uses the new unified agent internally.
    """
    
    def __init__(self, config: Dict[str, Any], workflow_factory: Optional[Callable] = None):
        super().__init__(config, workflow_factory)
        logger.info("✅ Using unified call center agent via compatibility wrapper")


# ============================================================================
# FIX 2: Enhanced Workflow Factory with Better Error Handling
# ============================================================================

def create_unified_voice_handler(config: Dict[str, Any]) -> UnifiedVoiceHandler:
    """Factory function to create unified voice handler with enhanced error handling."""
    
    def workflow_factory(client_data: Dict[str, Any]) -> CompiledGraph:
        """Create workflow for given client data with enhanced error handling."""
        logger.info(f"🏭 workflow_factory called with client_data keys: {list(client_data.keys()) if client_data else 'None'}")
        
        try:
            # Try to import required modules
            try:
                from ..agent.unified_workflow import create_unified_call_center_workflow
                logger.info("✅ Successfully imported create_unified_call_center_workflow")
            except ImportError as e:
                logger.error(f"❌ Failed to import unified workflow: {e}")
                # Try fallback to original system
                try:
                    from src.Agents.graph_call_center_agent import create_call_center_agent as create_unified_call_center_workflow
                    logger.info("✅ Using fallback to original call center agent")
                except ImportError as e2:
                    logger.error(f"❌ Fallback import also failed: {e2}")
                    raise ImportError(f"Cannot import workflow creator: {e}, fallback: {e2}")
            
            try:
                from langchain_ollama import ChatOllama
                logger.info("✅ Successfully imported ChatOllama")
            except ImportError as e:
                logger.error(f"❌ Failed to import ChatOllama: {e}")
                raise ImportError(f"ChatOllama not available: {e}")
            
            # Get model config
            llm_config = config.get('llm', {})
            model_name = llm_config.get('model_name', 'qwen2.5:14b-instruct')
            temperature = llm_config.get('temperature', 0)
            
            logger.info(f"🤖 Creating model: {model_name} with temperature: {temperature}")
            
            # Create model
            model = ChatOllama(model=model_name, temperature=temperature)
            logger.info("✅ Model created successfully")
            
            # Create workflow
            logger.info("🔧 Creating workflow...")
            workflow = create_unified_call_center_workflow(
                model=model,
                client_data=client_data,
                config=config,
                agent_name="Sarah"
            )
            
            if workflow:
                logger.info("✅ Workflow created successfully")
            else:
                logger.error("❌ Workflow creation returned None")
                
            return workflow
            
        except Exception as e:
            logger.error(f"❌ Failed to create workflow: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    logger.info("🏭 Creating UnifiedVoiceHandler with enhanced workflow factory")
    return UnifiedVoiceHandler(config, workflow_factory)


# ============================================================================
# FIX 3: Debug Helper Function
# ============================================================================

def debug_workflow_creation(config: Dict[str, Any], client_data: Dict[str, Any]):
    """Debug helper to test workflow creation independently."""
    print("=== DEBUGGING WORKFLOW CREATION ===")
    
    print(f"1. Config keys: {list(config.keys())}")
    print(f"2. Client data keys: {list(client_data.keys()) if client_data else 'None'}")
    
    # Test imports
    print("3. Testing imports...")
    try:
        from langchain_ollama import ChatOllama
        print("   ✅ ChatOllama import successful")
    except ImportError as e:
        print(f"   ❌ ChatOllama import failed: {e}")
        return False
    
    try:
        from src.Agents.unified_call_center_agent.agent.unified_workflow import create_unified_call_center_workflow
        print("   ✅ Unified workflow import successful")
    except ImportError as e:
        print(f"   ❌ Unified workflow import failed: {e}")
        try:
            from src.Agents.graph_call_center_agent import create_call_center_agent
            print("   ✅ Fallback to original workflow successful")
            create_unified_call_center_workflow = create_call_center_agent
        except ImportError as e2:
            print(f"   ❌ Fallback import also failed: {e2}")
            return False
    
    # Test model creation
    print("4. Testing model creation...")
    try:
        model = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)
        print("   ✅ Model created successfully")
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False
    
    # Test workflow creation
    print("5. Testing workflow creation...")
    try:
        workflow = create_unified_call_center_workflow(
            model=model,
            client_data=client_data,
            config=config
        )
        if workflow:
            print("   ✅ Workflow created successfully")
            return True
        else:
            print("   ❌ Workflow creation returned None")
            return False
    except Exception as e:
        print(f"   ❌ Workflow creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

