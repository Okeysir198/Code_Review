"""
Simplified Generic Voice Interaction Handler for any LangGraph workflow.
Supports STT, TTS, and message streaming with minimal complexity.
"""

import logging
import time
import uuid
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from fastrtc import AdditionalOutputs
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.STT import BaseSTTModel, create_stt_model
from src.TTS import BaseTTSModel, create_tts_model

logger = logging.getLogger(__name__)


class SimpleVoiceHandler:
    """
    Simplified voice interaction handler that works with any LangGraph workflow.
    Focus on core functionality: STT → Graph → TTS with message streaming.
    """
    
    def __init__(self, 
                 stt_config: Optional[Dict[str, Any]] = None,
                 tts_config: Optional[Dict[str, Any]] = None,
                 enable_stt: bool = True,
                 enable_tts: bool = True):
        """
        Initialize the simplified voice handler.
        
        Args:
            stt_config: STT model configuration
            tts_config: TTS model configuration  
            enable_stt: Whether to enable speech-to-text
            enable_tts: Whether to enable text-to-speech
        """
        self.enable_stt = enable_stt
        self.enable_tts = enable_tts
        
        # Initialize models
        self.stt_model = self._init_stt(stt_config) if enable_stt else None
        self.tts_model = self._init_tts(tts_config) if enable_tts else None
        
        # Message handlers for real-time streaming
        self.message_handlers: List[Callable[[str, str], None]] = []
        
        logger.info("SimpleVoiceHandler initialized")

    def _init_stt(self, config: Optional[Dict[str, Any]] = None) -> Optional[BaseSTTModel]:
        """Initialize STT model with error handling."""
        try:
            config = config or {"model_name": "nvidia/parakeet-tdt-0.6b-v2"}
            return create_stt_model(config)
        except Exception as e:
            logger.error(f"STT initialization failed: {e}")
            return None

    def _init_tts(self, config: Optional[Dict[str, Any]] = None) -> Optional[BaseTTSModel]:
        """Initialize TTS model with error handling."""
        try:
            config = config or {"model_name": "openai/tts-1"}
            return create_tts_model(config)
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            return None

    def _normalize_audio(self, audio_input: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray]:
        """Normalize audio to int16 format."""
        try:
            sample_rate, audio_array = audio_input
            
            if audio_array.dtype == np.float32:
                audio_array = np.clip(audio_array, -1.0, 1.0)
                audio_array = (audio_array * 32767).astype(np.int16)
            elif audio_array.dtype != np.int16:
                audio_array = audio_array.astype(np.int16)
            
            return (sample_rate, audio_array)
        except Exception as e:
            logger.error(f"Audio normalization error: {e}")
            return (16000, np.array([0], dtype=np.int16))

    def _stream_message(self, role: str, content: str):
        """Stream message to all registered handlers."""
        for handler in self.message_handlers:
            try:
                handler(role, content)
            except Exception as e:
                logger.error(f"Message handler error: {e}")

    def add_message_handler(self, handler: Callable[[str, str], None]):
        """Add a message handler for real-time streaming."""
        self.message_handlers.append(handler)

    def process_with_graph(self, 
                          user_input: str, 
                          graph: CompiledGraph,
                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process user input through any LangGraph workflow.
        
        Args:
            user_input: User's text input
            graph: Any compiled LangGraph workflow
            config: Configuration for the graph (thread_id, etc.)
            
        Returns:
            Dictionary with messages and metadata
        """
        try:
            # Default config with thread_id
            if config is None:
                config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            
            # Prepare input for the graph
            graph_input = {"messages": [HumanMessage(content=user_input)]}
            
            # Stream user message
            self._stream_message("user", user_input)
            
            # Process through graph
            all_messages = []
            result = None
            
            for event in graph.stream(graph_input, config=config, stream_mode="values"):
                if event and isinstance(event, dict) and "messages" in event:
                    result = event
                    
                    # Process latest message
                    if event["messages"]:
                        latest_message = event["messages"][-1]
                        
                        # Handle AI messages
                        if isinstance(latest_message, AIMessage) and latest_message.content:
                            self._stream_message("assistant", latest_message.content)
                            all_messages.append({
                                "role": "assistant",
                                "content": latest_message.content,
                                "timestamp": time.time()
                            })
                        
                        # Handle tool calls
                        if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                            for tool_call in latest_message.tool_calls:
                                tool_name = getattr(tool_call, "name", "unknown")
                                tool_args = getattr(tool_call, "args", {})
                                tool_msg = f"Calling {tool_name}({tool_args})"
                                self._stream_message("tool", tool_msg)
                                all_messages.append({
                                    "role": "tool_call",
                                    "content": tool_msg,
                                    "timestamp": time.time()
                                })
                        
                        # Handle tool responses
                        if isinstance(latest_message, ToolMessage):
                            tool_name = getattr(latest_message, "name", "unknown")
                            content = getattr(latest_message, "content", "")
                            response_msg = f"{tool_name} responded: {content}"
                            self._stream_message("tool", response_msg)
                            all_messages.append({
                                "role": "tool_response", 
                                "content": response_msg,
                                "timestamp": time.time()
                            })
            
            return {
                "success": True,
                "result": result,
                "all_messages": all_messages,
                "final_response": self._extract_final_response(result)
            }
            
        except Exception as e:
            logger.error(f"Graph processing error: {e}")
            error_msg = f"Error processing request: {str(e)}"
            self._stream_message("error", error_msg)
            return {
                "success": False,
                "error": error_msg,
                "all_messages": [{"role": "error", "content": error_msg, "timestamp": time.time()}]
            }

    def _extract_final_response(self, result: Optional[Dict[str, Any]]) -> str:
        """Extract the final AI response from graph result."""
        if not result or "messages" not in result:
            return "I apologize, but I couldn't generate a response."
        
        # Find the last AI message
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        
        return "I apologize, but I couldn't generate a response."

    def process_text_input(self, 
                          text_input: str,
                          graph: CompiledGraph,
                          config: Optional[Dict[str, Any]] = None,
                          chatbot: Optional[List[Dict[str, str]]] = None) -> Generator:
        """
        Process text input and optionally generate TTS.
        
        Args:
            text_input: User's text input
            graph: Any compiled LangGraph workflow
            config: Graph configuration
            chatbot: Current chatbot state for UI
            
        Yields:
            Audio chunks (if TTS enabled) and updated chatbot state
        """
        try:
            chatbot = chatbot or []
            
            # Add user message to chatbot
            chatbot.append({"role": "user", "content": text_input})
            yield None, chatbot, ""
            
            # Process through graph
            result = self.process_with_graph(text_input, graph, config)
            
            if result["success"]:
                # Add all messages to chatbot
                for msg in result["all_messages"]:
                    if msg["role"] != "user":  # Skip user (already added)
                        chatbot.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                yield None, chatbot, ""
                
                # Generate TTS for final AI response
                if self.enable_tts and self.tts_model:
                    final_response = result["final_response"]
                    if final_response and final_response.strip():
                        try:
                            for audio_chunk in self.tts_model.stream_text_to_speech(final_response):
                                yield self._normalize_audio(audio_chunk), chatbot, ""
                        except Exception as e:
                            logger.error(f"TTS generation failed: {e}")
            else:
                # Add error to chatbot
                chatbot.append({"role": "error", "content": result["error"]})
                yield None, chatbot, ""
                
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            chatbot.append({"role": "error", "content": f"Processing error: {str(e)}"})
            yield None, chatbot, ""

    def process_audio_input(self,
                           audio_input: Tuple[int, np.ndarray],
                           graph: CompiledGraph,
                           config: Optional[Dict[str, Any]] = None,
                           chatbot: Optional[List[Dict[str, str]]] = None) -> Generator:
        """
        Process audio input: STT → Graph → TTS.
        
        Args:
            audio_input: Tuple of (sample_rate, audio_array)
            graph: Any compiled LangGraph workflow
            config: Graph configuration
            chatbot: Current chatbot state for UI
            
        Yields:
            Audio chunks and updated chatbot state
        """
        try:
            chatbot = chatbot or []
            sample_rate, audio_array = audio_input
            
            # Validate audio
            if not audio_array.size or not self.enable_stt or not self.stt_model:
                yield (sample_rate, np.array([0], dtype=np.int16))
                yield AdditionalOutputs(chatbot)
                return
            
            # Transcribe audio
            try:
                stt_result = self.stt_model.transcribe(audio_input)
                transcription = (stt_result.get("text", "") if isinstance(stt_result, dict) 
                               else str(stt_result)).strip()
            except Exception as e:
                logger.error(f"STT failed: {e}")
                yield (sample_rate, np.array([0], dtype=np.int16))
                yield AdditionalOutputs(chatbot)
                return
            
            if not transcription:
                yield (sample_rate, np.array([0], dtype=np.int16))
                yield AdditionalOutputs(chatbot)
                return
            
            # Add user message to chatbot
            chatbot.append({"role": "user", "content": transcription})
            yield AdditionalOutputs(chatbot)
            
            # Process through graph
            result = self.process_with_graph(transcription, graph, config)
            
            if result["success"]:
                # Add all messages to chatbot
                for msg in result["all_messages"]:
                    if msg["role"] != "user":  # Skip user (already added)
                        chatbot.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                yield AdditionalOutputs(chatbot)
                
                # Generate TTS for final AI response
                if self.enable_tts and self.tts_model:
                    final_response = result["final_response"]
                    if final_response and final_response.strip():
                        try:
                            for audio_chunk in self.tts_model.stream_text_to_speech(final_response):
                                yield self._normalize_audio(audio_chunk)
                        except Exception as e:
                            logger.error(f"TTS generation failed: {e}")
                            yield (sample_rate, np.array([0], dtype=np.int16))
                else:
                    yield (sample_rate, np.array([0], dtype=np.int16))
            else:
                # Add error to chatbot
                chatbot.append({"role": "error", "content": result["error"]})
                yield AdditionalOutputs(chatbot)
                yield (sample_rate, np.array([0], dtype=np.int16))
                
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            chatbot.append({"role": "error", "content": f"Audio processing error: {str(e)}"})
            yield (sample_rate, np.array([0], dtype=np.int16))
            yield AdditionalOutputs(chatbot)


# Example usage with any graph:
def create_simple_voice_app(graph: CompiledGraph):
    """
    Example of how to use the simplified voice handler with any graph.
    """
    import gradio as gr
    
    # Initialize handler
    handler = SimpleVoiceHandler(
        enable_stt=True,
        enable_tts=True
    )
    
    # Optional: Add message handler for real-time streaming
    def print_message(role: str, content: str):
        print(f"[{role.upper()}]: {content}")
    
    handler.add_message_handler(print_message)
    
    with gr.Blocks() as demo:
        gr.Markdown("# Generic Voice Chat with Any LangGraph")
        
        chatbot = gr.Chatbot(label="Conversation")
        
        with gr.Row():
            # Text input
            text_input = gr.Textbox(label="Type your message", scale=4)
            text_btn = gr.Button("Send", scale=1)
            
        # Audio input/output
        audio_input = gr.Audio(label="Speak", sources=["microphone"])
        audio_output = gr.Audio(label="AI Response", autoplay=True)
        
        # Text processing
        def process_text(text, chat_history):
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            for result in handler.process_text_input(text, graph, config, chat_history):
                if result[0] is None:  # Text update
                    yield result[1], "", result[0]  # chatbot, clear input, audio
                else:  # Audio update
                    yield result[1], "", result[0]  # chatbot, clear input, audio
        
        text_btn.click(
            process_text,
            inputs=[text_input, chatbot],
            outputs=[chatbot, text_input, audio_output]
        )
        
        # Audio processing
        def process_audio(audio, chat_history):
            if audio is None:
                return chat_history, None
            
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            final_chatbot = chat_history
            final_audio = None
            
            for result in handler.process_audio_input(audio, graph, config, chat_history):
                if isinstance(result, AdditionalOutputs):
                    final_chatbot = result
                else:
                    final_audio = result
            
            return final_chatbot, final_audio
        
        audio_input.change(
            process_audio,
            inputs=[audio_input, chatbot],
            outputs=[chatbot, audio_output]
        )
    
    return demo