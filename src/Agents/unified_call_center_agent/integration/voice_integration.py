"""Enhanced Voice interaction handler for AI-powered conversational agents with message streaming."""

import logging
import uuid
import json
from typing import Dict, Any, Optional, Tuple, List, Generator, Union, Set, Callable
import numpy as np
from fastrtc import AdditionalOutputs
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.STT import create_stt_model, BaseSTTModel
from src.TTS import create_tts_model, BaseTTSModel

logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes."""
    YELLOW = "\x1B[33m"
    GREEN = "\x1B[32m"
    CYAN = "\x1B[36m"
    MAGENTA = "\x1B[35m"
    BLUE = "\x1B[34m"
    RESET = "\x1B[0m"
    BOLD = "\x1B[1m"


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


class VoiceInteractionHandler:
    """Enhanced voice interaction handler with message streaming."""
    
    def __init__(self, config: Dict[str, Any], workflow_factory: Optional[Callable] = None):
        self.config = self._setup_config(config or {})
        self.workflow_factory = workflow_factory
        self.message_streamer = MessageStreamer()
        self._setup_logging()
        
        # Client data caching
        self.cached_client_data = None
        self.cached_user_id = None
        self.cached_workflow = None
        
        # Initialize models
        self.stt_model = self._init_stt() if self.config['configurable'].get('enable_stt_model') else None
        self.tts_model = self._init_tts() if self.config['configurable'].get('enable_tts_model') else None

    def add_message_handler(self, handler: Callable[[str, str], None]):
        """Add external message handler for real-time streaming."""
        self.message_streamer.add_handler(handler)

    def update_client_data(self, user_id: str, client_data: Dict[str, Any]):
        """Update cached client data and workflow."""
        try:
            if user_id != self.cached_user_id or not self.cached_workflow:
                logger.info(f"Updating client data cache for user_id: {user_id}")
                
                self.cached_user_id = user_id
                self.cached_client_data = client_data
                
                # Create new workflow with cached data
                if self.workflow_factory and client_data:
                    self.cached_workflow = self.workflow_factory(client_data)
                    logger.info(f"Created new workflow for user_id: {user_id}")
                else:
                    self.cached_workflow = None
                    logger.warning(f"No workflow factory or client data for user_id: {user_id}")
                    
        except Exception as e:
            logger.error(f"Error updating client data for user_id {user_id}: {e}")
            self.cached_workflow = None

    def get_current_workflow(self) -> Optional[CompiledGraph]:
        """Get the current cached workflow."""
        return self.cached_workflow

    def _setup_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Setup configuration with defaults."""
        config.setdefault('stt', {})
        config.setdefault('tts', {})
        config.setdefault('logging', {'level': 'error', 'console_output': True})
        config.setdefault('configurable', {
            'thread_id': str(uuid.uuid4()),
            'enable_stt_model': True,
            'enable_tts_model': True
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
        
        level = level_map.get(self.config['logging'].get('level', 'error').lower(), logging.ERROR)
        logger.setLevel(level)
        
        if self.config['logging'].get('console_output', True) and not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logger.addHandler(handler)

    def _init_stt(self) -> BaseSTTModel:
        """Initialize STT model."""
        try:
            return create_stt_model(self.config['stt'])
        except Exception as e:
            logger.error(f"STT init failed: {e}")
            raise

    def _init_tts(self) -> BaseTTSModel:
        """Initialize TTS model."""
        try:
            return create_tts_model(self.config['tts'])
        except Exception as e:
            logger.error(f"TTS init failed: {e}")
            raise

    def _print(self, label: str, text: str, color: str):
        """Print formatted message."""
        print(f"{Colors.BOLD}{color}[{label}]:{Colors.RESET} {color}{text}{Colors.RESET}")

    def _print_tool_response(self, tool_name: str, content: Any):
        """Print tool response."""
        print(f"{Colors.BOLD}{Colors.MAGENTA}[Tool Response - {tool_name}]:{Colors.RESET}")
        try:
            if isinstance(content, str):
                formatted = json.dumps(json.loads(content), indent=2)
            elif isinstance(content, dict):
                formatted = json.dumps(content, indent=2)
            else:
                formatted = str(content)
        except:
            formatted = str(content)
        print(f"{Colors.MAGENTA}{formatted}{Colors.RESET}")

    def _process_tool_messages(self, message, tracked_calls: Set[str]):
        """Process tool messages with streaming (no duplication)."""
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_id = getattr(tool_call, "id", str(id(tool_call)))
                tool_name = getattr(tool_call, "name", "unknown")
                tool_args = getattr(tool_call, "args", {})
                call_id = f"{tool_name}:{tool_id}"
                
                if call_id not in tracked_calls:
                    tool_msg = f"{tool_name}({tool_args})"
                    self._print("Tool Call", tool_msg, Colors.BLUE)
                    tracked_calls.add(call_id)
        
        if isinstance(message, ToolMessage):
            tool_name = getattr(message, "name", "Unknown Tool")
            content = getattr(message, "content", None)
            if content:
                self._print_tool_response(tool_name, content)
                # Only stream tool response once per message
                tool_msg_id = f"tool_response_{id(message)}"
                if tool_msg_id not in tracked_calls:
                    try:
                        if isinstance(content, str) and len(content) > 200:
                            truncated = content[:200] + "..."
                            self.message_streamer.stream_message("tool", f"RESPONSE: {tool_name} -> {truncated}")
                        else:
                            self.message_streamer.stream_message("tool", f"RESPONSE: {tool_name} -> {content}")
                        tracked_calls.add(tool_msg_id)
                    except Exception as e:
                        self.message_streamer.stream_message("tool", f"RESPONSE: {tool_name} -> [Error: {e}]")

    def _extract_ai_response(self, workflow_result: Dict[str, Any]) -> str:
        """Extract AI response from workflow result."""
        if "error" in workflow_result:
            return workflow_result["error"]
        
        messages = workflow_result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
                return msg.content
        
        return "No AI response found."

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

    def process_message(self, user_message: str, workflow: CompiledGraph = None) -> Dict[str, Any]:
        """Process message through workflow with enhanced streaming (no duplication)."""
        # Use cached workflow if no workflow provided
        if workflow is None:
            workflow = self.get_current_workflow()
        
        if not workflow:
            return {"messages": [], "error": "No workflow available. Please select a client first."}
        
        try:
            workflow_input = {"messages": [HumanMessage(content=user_message)]}
            config = {"configurable": self.config.get('configurable', {})}
            
            tracked_calls = set()
            tracked_messages = set()  # Track streamed messages to prevent duplication
            final_result = None
            
            # Stream user message only once
            user_msg_id = f"user_{hash(user_message)}"
            if user_msg_id not in tracked_messages:
                self.message_streamer.stream_message("user", user_message)
                tracked_messages.add(user_msg_id)
            
            for event in workflow.stream(workflow_input, config=config, stream_mode="values"):
                if event and isinstance(event, dict) and "messages" in event:
                    final_result = event
                    
                    # Process only new messages to avoid duplication
                    if event["messages"]:
                        latest_message = event["messages"][-1]
                        
                        # Process tool messages
                        self._process_tool_messages(latest_message, tracked_calls)
                        
                        # Stream AI messages only once
                        if isinstance(latest_message, AIMessage) and latest_message.content:
                            ai_msg_id = f"ai_{id(latest_message)}"
                            if ai_msg_id not in tracked_messages:
                                self.message_streamer.stream_message("ai", latest_message.content)
                                tracked_messages.add(ai_msg_id)
            
            return final_result or {"messages": [], "error": "No valid events received"}
            
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            self.message_streamer.stream_message("error", f"Workflow error: {e}")
            return {"messages": [], "error": f"Workflow execution error: {str(e)}"}

    def process_text_input(self, text_input: str, workflow: CompiledGraph = None,
                          gradio_chatbot: Optional[List[Dict[str, str]]] = None,
                          thread_id: Optional[Union[str, int]] = None) -> Generator:
        """Process text input with streaming response."""
        try:
            chatbot = gradio_chatbot or []
            
            # Set thread ID
            self.config['configurable']["thread_id"] = str(thread_id or uuid.uuid4())
            
            # Add user message
            self._print("Text Input", text_input, Colors.YELLOW)
            chatbot.append({"role": "user", "content": text_input})
            yield None, chatbot, ""

            # Process and get response with cached workflow
            workflow_result = self.process_message(text_input, workflow)
            response = self._extract_ai_response(workflow_result)
            
            # Add AI response
            self._print("AI Response", response, Colors.GREEN)
            chatbot.append({"role": "assistant", "content": response})
            yield None, chatbot, ""
            
            # Generate TTS if enabled
            if self.tts_model and self.config['configurable'].get('enable_tts_model') and response.strip():
                for audio_chunk in self.tts_model.stream_text_to_speech(response):
                    yield self._normalize_audio(audio_chunk), chatbot, ""
                
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            chatbot.append({"role": "assistant", "content": "Sorry, an error occurred."})
            yield None, chatbot, ""

    def process_audio_input(self, audio_input: Tuple[int, np.ndarray], workflow: CompiledGraph = None,
                           gradio_chatbot: Optional[List[Dict[str, str]]] = None,
                           thread_id: Optional[Union[str, int]] = None) -> Generator:
        """Process audio input with enhanced streaming response."""
        try:
            chatbot = gradio_chatbot or []
            sample_rate = audio_input[0]
            
            # Check if audio is valid
            if (not audio_input[1].size or not self.stt_model or 
                not self.config['configurable'].get('enable_stt_model')):
                yield (sample_rate, np.array([], dtype=np.int16))
                yield AdditionalOutputs(self._format_chatbot(chatbot))
                return
            
            # Transcribe audio
            stt_result = self.stt_model.transcribe(audio_input)
            stt_text = (stt_result.get("text", "") if isinstance(stt_result, dict) 
                       else str(stt_result)).strip()
            
            if not stt_text:
                yield (sample_rate, np.array([], dtype=np.int16))
                yield AdditionalOutputs(self._format_chatbot(chatbot))
                return
            
            # Process conversation
            self._print("Voice Input", stt_text, Colors.YELLOW)
            chatbot.append({"role": "user", "content": stt_text})
            
            # Set thread ID and get response with cached workflow
            self.config['configurable']["thread_id"] = str(thread_id or uuid.uuid4())
            workflow_result = self.process_message(stt_text, workflow)
            response = self._extract_ai_response(workflow_result)
            
            self._print("AI Response", response, Colors.GREEN)
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
            logger.error(f"Audio processing error: {e}")
            chatbot.append({"role": "assistant", "content": "Sorry, an error occurred."})
            yield (16000, np.array([0], dtype=np.int16))
            yield AdditionalOutputs(self._format_chatbot(chatbot))

    def set_log_level(self, level: str):
        """Set logging level at runtime."""
        self.config['logging']['level'] = level
        self._setup_logging()