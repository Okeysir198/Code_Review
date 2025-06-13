"""
LiveKit Voice Interaction Handler - Self-Hosted Version
Replaces FastRTC with LiveKit while maintaining all existing functionality
No API keys required - connects directly to self-hosted LiveKit server
"""

import asyncio
import hashlib
import json
import logging
import random
import time
import uuid
from collections import defaultdict
from functools import lru_cache
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np
from livekit import rtc, api
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    stt,
    tts,
    AudioSource
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import silero

from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.STT import BaseSTTModel, create_stt_model
from src.TTS import BaseTTSModel, create_tts_model

logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for console output."""
    YELLOW = "\x1B[33m"
    GREEN = "\x1B[32m"
    CYAN = "\x1B[36m"
    MAGENTA = "\x1B[35m"
    BLUE = "\x1B[34m"
    RED = "\x1B[31m"
    RESET = "\x1B[0m"
    BOLD = "\x1B[1m"


class PerformanceMonitor:
    """Monitor and track system performance metrics."""
    
    def __init__(self, max_history: int = 100):
        self.metrics = defaultdict(list)
        self.max_history = max_history
        self.start_times = {}
    
    def start_timer(self, operation: str) -> str:
        """Start timing an operation."""
        timer_id = f"{operation}_{uuid.uuid4().hex[:8]}"
        self.start_times[timer_id] = time.time()
        return timer_id
    
    def end_timer(self, timer_id: str) -> float:
        """End timing and record duration."""
        if timer_id in self.start_times:
            duration = time.time() - self.start_times.pop(timer_id)
            operation = timer_id.split('_')[0]
            self.track_metric(f"{operation}_latency", duration)
            return duration
        return 0.0
    
    def track_metric(self, metric_name: str, value: float):
        """Track a metric value."""
        self.metrics[metric_name].append(value)
        if len(self.metrics[metric_name]) > self.max_history:
            self.metrics[metric_name].pop(0)
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        return summary


class MessageStreamer:
    """Enhanced message streaming with deduplication and filtering."""
    
    def __init__(self):
        self.handlers: Set[Callable] = set()
        self.processed_ids: Set[str] = set()
        self.max_processed = 1000
    
    def add_handler(self, handler: Callable):
        """Add a message handler."""
        self.handlers.add(handler)
    
    def remove_handler(self, handler: Callable):
        """Remove a message handler."""
        self.handlers.discard(handler)
    
    def stream_message(self, message: Dict[str, Any]):
        """Stream message to all handlers with deduplication."""
        msg_id = self._get_message_id(message)
        
        if msg_id in self.processed_ids:
            return
        
        self.processed_ids.add(msg_id)
        
        if len(self.processed_ids) > self.max_processed:
            self.processed_ids = set(list(self.processed_ids)[-self.max_processed//2:])
        
        for handler in self.handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
    
    def _get_message_id(self, message: Dict[str, Any]) -> str:
        """Generate unique message ID."""
        content = str(message.get('content', ''))
        role = str(message.get('role', ''))
        return hashlib.md5(f"{role}:{content}".encode()).hexdigest()
    
    def clear_handlers(self):
        """Clear all handlers."""
        self.handlers.clear()


class CustomSTTAdapter:
    """Adapter to use existing STT models with LiveKit."""
    
    def __init__(self, stt_model: BaseSTTModel):
        self.stt_model = stt_model
    
    async def recognize(self, audio_data: rtc.AudioFrame) -> stt.SpeechEvent:
        """Convert audio frame to text using custom STT."""
        # Convert AudioFrame to numpy array
        sample_rate = audio_data.sample_rate
        audio_array = np.frombuffer(audio_data.data, dtype=np.int16)
        
        # Use existing STT model
        text = self.stt_model.transcribe((sample_rate, audio_array))
        
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language="en")]
        )


class CustomTTSAdapter:
    """Adapter to use existing TTS models with LiveKit."""
    
    def __init__(self, tts_model: BaseTTSModel):
        self.tts_model = tts_model
    
    async def synthesize(self, text: str) -> AsyncGenerator[rtc.AudioFrame, None]:
        """Convert text to audio using custom TTS."""
        # Use existing TTS model
        for audio_chunk in self.tts_model.stream_tts(text):
            sample_rate, audio_array = audio_chunk
            
            # Convert to AudioFrame
            frame = rtc.AudioFrame.create(
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(audio_array)
            )
            
            # Copy audio data
            frame_data = np.frombuffer(frame.data, dtype=np.int16)
            np.copyto(frame_data, audio_array.flatten())
            
            yield frame


class LiveKitVoiceInteractionHandler:
    """
    LiveKit-based voice interaction handler with self-hosted server support.
    Maintains compatibility with existing VoiceInteractionHandler interface.
    """
    
    def __init__(self, config: Dict[str, Any], workflow_factory: Optional[Callable] = None):
        # Core configuration
        self.config = self._setup_config(config or {})
        self.workflow_factory = workflow_factory
        
        # Enhanced components (matching original)
        self.message_streamer = MessageStreamer()
        self.performance_monitor = PerformanceMonitor()
        
        # Setup logging
        self._setup_logging()
        
        # Client data caching with TTL
        self.cached_client_data = None
        self.cached_user_id = None
        self.cached_workflow = None
        self.cache_timestamp = 0
        self.cache_ttl = 3600  # 1 hour
        
        # Performance optimization
        self.request_cache = {}
        self.processing_lock = asyncio.Lock()
        self.concurrent_limit = 3
        self.active_requests = 0
        
        # Fallback responses for error scenarios
        self.fallback_responses = [
            "I'm having trouble processing that. Could you please repeat?",
            "Sorry, there seems to be a technical issue. Let me try again.",
            "I need a moment to process. Please hold on.",
            "Could you please rephrase that? I didn't catch it clearly."
        ]
        
        # LiveKit components
        self.room: Optional[rtc.Room] = None
        self.voice_assistant: Optional[VoiceAssistant] = None
        self.participant: Optional[rtc.RemoteParticipant] = None
        
        # Audio processing settings
        self.vad = silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.3,
            padding_duration=0.1,
            sample_rate=16000,
            activation_threshold=0.6
        )
        
        # Initialize models with LiveKit adapters
        self.stt_adapter = None
        self.tts_adapter = None
        if self.config['configurable'].get('enable_stt_model'):
            stt_model = self._init_stt()
            self.stt_adapter = CustomSTTAdapter(stt_model)
        if self.config['configurable'].get('enable_tts_model'):
            tts_model = self._init_tts()
            self.tts_adapter = CustomTTSAdapter(tts_model)
        
        logger.info("LiveKitVoiceInteractionHandler initialized successfully")
    
    def _setup_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Setup configuration with production defaults."""
        config.setdefault('stt', {})
        config.setdefault('tts', {})
        config.setdefault('logging', {'level': 'info', 'console_output': True})
        config.setdefault('configurable', {
            'thread_id': str(uuid.uuid4()),
            'enable_stt_model': True,
            'enable_tts_model': True
        })
        config.setdefault('livekit', {
            'url': 'ws://localhost:7880',  # Self-hosted LiveKit server
            'use_cloud': False,  # No cloud API keys needed
            'room_name': f'voice-chat-{uuid.uuid4().hex[:8]}',
            'participant_name': 'AI-Assistant'
        })
        config.setdefault('audio', {
            'sample_rate': 16000,
            'channels': 1,
            'noise_suppression': True,
            'echo_cancellation': True,
            'auto_gain_control': True
        })
        return config
    
    def _setup_logging(self):
        """Configure production logging."""
        level_map = {
            'none': logging.CRITICAL + 1,
            'error': logging.ERROR,
            'warning': logging.WARNING,
            'info': logging.INFO,
            'debug': logging.DEBUG
        }
        
        level = level_map.get(self.config['logging'].get('level', 'info').lower(), logging.INFO)
        logger.setLevel(level)
    
    def _init_stt(self) -> BaseSTTModel:
        """Initialize STT model with error handling."""
        try:
            timer_id = self.performance_monitor.start_timer("stt_init")
            model = create_stt_model(self.config['stt'])
            self.performance_monitor.end_timer(timer_id)
            logger.info("STT model initialized successfully")
            return model
        except Exception as e:
            logger.error(f"STT initialization failed: {e}")
            raise
    
    def _init_tts(self) -> BaseTTSModel:
        """Initialize TTS model with error handling."""
        try:
            timer_id = self.performance_monitor.start_timer("tts_init")
            model = create_tts_model(self.config['tts'])
            self.performance_monitor.end_timer(timer_id)
            logger.info("TTS model initialized successfully")
            return model
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            raise
    
    async def connect_to_room(self, room_name: Optional[str] = None) -> str:
        """
        Connect to LiveKit room (self-hosted server).
        Returns the room name for client connection.
        """
        try:
            # Use provided room name or generate one
            if not room_name:
                room_name = self.config['livekit']['room_name']
            
            # Create room instance
            self.room = rtc.Room()
            
            # For self-hosted server, we don't need API keys
            # Just connect directly with a participant identity
            url = self.config['livekit']['url']
            token = self._generate_self_hosted_token(room_name)
            
            # Connect to room
            await self.room.connect(url, token)
            
            # Setup voice assistant
            await self._setup_voice_assistant()
            
            # Set up event handlers
            self.room.on("participant_connected", self._on_participant_connected)
            self.room.on("participant_disconnected", self._on_participant_disconnected)
            self.room.on("track_subscribed", self._on_track_subscribed)
            
            logger.info(f"Connected to LiveKit room: {room_name}")
            return room_name
            
        except Exception as e:
            logger.error(f"Failed to connect to LiveKit room: {e}")
            raise
    
    def _generate_self_hosted_token(self, room_name: str) -> str:
        """
        Generate a simple token for self-hosted LiveKit server.
        In production, this should use proper JWT signing.
        """
        # For self-hosted development, LiveKit can work with simple tokens
        # This is NOT secure for production - use proper JWT signing
        participant_name = self.config['livekit']['participant_name']
        
        # Create a basic token structure
        # In production, use the LiveKit SDK's AccessToken class
        token_data = {
            "identity": participant_name,
            "room": room_name,
            "video": {"room": "join", "roomJoin": True},
            "metadata": json.dumps({"role": "assistant"})
        }
        
        # For self-hosted dev server, often a simple base64 encoding works
        # IMPORTANT: Replace with proper JWT signing for production
        import base64
        token = base64.b64encode(json.dumps(token_data).encode()).decode()
        
        return token
    
    async def _setup_voice_assistant(self):
        """Initialize voice assistant with custom STT/TTS or LiveKit defaults."""
        # Use custom adapters if available, otherwise LiveKit defaults
        if self.stt_adapter:
            stt_plugin = self.stt_adapter
        else:
            # Fallback to LiveKit's built-in STT
            stt_plugin = stt.StreamAdapter(
                stt=stt.deepgram.STT(),  # or another provider
                vad=self.vad
            )
        
        if self.tts_adapter:
            tts_plugin = self.tts_adapter
        else:
            # Fallback to LiveKit's built-in TTS
            tts_plugin = tts.StreamAdapter(
                tts=tts.elevenlabs.TTS()  # or another provider
            )
        
        # Create custom LLM handler that uses LangGraph workflow
        llm_handler = LangGraphLLMHandler(
            workflow_factory=self.workflow_factory,
            message_streamer=self.message_streamer,
            performance_monitor=self.performance_monitor
        )
        
        # Create voice assistant
        self.voice_assistant = VoiceAssistant(
            vad=self.vad,
            stt=stt_plugin,
            llm=llm_handler,
            tts=tts_plugin,
            chat_ctx=None,  # Will be set when participant connects
            fnc_ctx=None,   # Function calling context if needed
            allow_interruptions=True,
            interrupt_speech_duration=0.5,
            interrupt_min_words=3,
            min_endpointing_delay=0.5,
            preemptive_synthesis=True,
            transcription=stt_plugin  # Use same STT for transcription
        )
        
        # Set up event handlers
        self.voice_assistant.on("user_speech_started", self._on_user_speech_started)
        self.voice_assistant.on("user_speech_ended", self._on_user_speech_ended)
        self.voice_assistant.on("agent_speech_started", self._on_agent_speech_started)
        self.voice_assistant.on("agent_speech_ended", self._on_agent_speech_ended)
        self.voice_assistant.on("user_speech_committed", self._on_user_speech_committed)
        self.voice_assistant.on("agent_speech_committed", self._on_agent_speech_committed)
    
    async def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        """Handle participant connection."""
        logger.info(f"Participant connected: {participant.identity}")
        self.participant = participant
        
        # Start voice assistant for this participant
        if self.voice_assistant:
            self.voice_assistant.start(self.room, participant)
    
    async def _on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """Handle participant disconnection."""
        logger.info(f"Participant disconnected: {participant.identity}")
        if participant == self.participant:
            self.participant = None
    
    async def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Handle track subscription."""
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Subscribed to audio track from {participant.identity}")
    
    async def _on_user_speech_started(self):
        """Handle user speech start."""
        logger.debug("User started speaking")
        self.message_streamer.stream_message({
            "role": "system",
            "content": "User started speaking",
            "timestamp": time.time()
        })
    
    async def _on_user_speech_ended(self):
        """Handle user speech end."""
        logger.debug("User stopped speaking")
    
    async def _on_agent_speech_started(self):
        """Handle agent speech start."""
        logger.debug("Agent started speaking")
    
    async def _on_agent_speech_ended(self):
        """Handle agent speech end."""
        logger.debug("Agent stopped speaking")
    
    async def _on_user_speech_committed(self, text: str):
        """Handle committed user speech."""
        self.message_streamer.stream_message({
            "role": "user",
            "content": text,
            "timestamp": time.time()
        })
    
    async def _on_agent_speech_committed(self, text: str):
        """Handle committed agent speech."""
        self.message_streamer.stream_message({
            "role": "assistant",
            "content": text,
            "timestamp": time.time()
        })
    
    def set_client_data(self, client_data: Dict[str, Any], user_id: str):
        """Set client data and create/cache workflow."""
        timer_id = self.performance_monitor.start_timer("set_client_data")
        
        try:
            self.cached_client_data = client_data
            self.cached_user_id = user_id
            self.cache_timestamp = time.time()
            
            # Create workflow if factory provided
            if self.workflow_factory:
                self.cached_workflow = self.workflow_factory(client_data)
                logger.info(f"Workflow created for user: {user_id}")
            
            self.performance_monitor.end_timer(timer_id)
            
        except Exception as e:
            self.performance_monitor.end_timer(timer_id)
            logger.error(f"Failed to set client data: {e}")
            raise
    
    def get_current_workflow(self) -> Optional[CompiledGraph]:
        """Get current workflow, checking cache validity."""
        if not self.cached_workflow:
            return None
        
        # Check cache age
        if time.time() - self.cache_timestamp > self.cache_ttl:
            logger.warning("Workflow cache expired")
            return None
        
        return self.cached_workflow
    
    def add_message_handler(self, handler: Callable):
        """Add a handler for message streaming."""
        self.message_streamer.add_handler(handler)
        logger.debug("Added message handler")
    
    def remove_message_handler(self, handler: Callable):
        """Remove a message handler."""
        self.message_streamer.remove_handler(handler)
        logger.debug("Removed message handler")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'performance': self.performance_monitor.get_metrics_summary(),
            'cache_stats': {
                'client_cache_age': time.time() - self.cache_timestamp if self.cache_timestamp else 0,
                'request_cache_size': len(self.request_cache),
                'active_requests': self.active_requests
            }
        }
    
    def set_log_level(self, level: str):
        """Change logging level dynamically."""
        self.config['logging']['level'] = level
        self._setup_logging()
        logger.info(f"Log level changed to: {level}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for monitoring."""
        return {
            "timestamp": time.time(),
            "performance_metrics": self.performance_monitor.get_metrics_summary(),
            "cache_info": {
                "client_cache_valid": self.cached_workflow is not None,
                "client_cache_age": time.time() - self.cache_timestamp if self.cache_timestamp else 0,
                "request_cache_size": len(self.request_cache),
            },
            "configuration": {
                "stt_enabled": self.config['configurable'].get('enable_stt_model', False),
                "tts_enabled": self.config['configurable'].get('enable_tts_model', False),
                "log_level": self.config['logging'].get('level', 'info'),
                "concurrent_limit": self.concurrent_limit,
                "active_requests": self.active_requests,
            },
            "livekit_status": {
                "connected": self.room is not None and self.room.connection_state == "connected",
                "room_name": self.room.name if self.room else None,
                "participant_count": len(self.room.participants) if self.room else 0
            }
        }
    
    async def disconnect(self):
        """Disconnect from LiveKit room and cleanup."""
        try:
            if self.voice_assistant:
                await self.voice_assistant.stop()
            
            if self.room:
                await self.room.disconnect()
            
            logger.info("Disconnected from LiveKit room")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    def cleanup_resources(self):
        """Clean up resources and connections."""
        try:
            logger.info("Cleaning up LiveKitVoiceInteractionHandler resources...")
            
            # Clear caches
            self.request_cache.clear()
            self.cached_client_data = None
            self.cached_workflow = None
            
            # Clear message streamer
            self.message_streamer.clear_handlers()
            
            # Reset counters
            self.active_requests = 0
            
            logger.info("Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class LangGraphLLMHandler:
    """
    Custom LLM handler that integrates LangGraph workflow with LiveKit.
    """
    
    def __init__(
        self,
        workflow_factory: Optional[Callable] = None,
        message_streamer: Optional[MessageStreamer] = None,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        self.workflow_factory = workflow_factory
        self.message_streamer = message_streamer
        self.performance_monitor = performance_monitor
        self.current_workflow = None
        self.conversation_history = []
    
    async def chat(
        self,
        chat_ctx: llm.ChatContext,
        fnc_ctx: Optional[llm.FunctionContext] = None,
    ) -> llm.LLMStream:
        """
        Process chat through LangGraph workflow.
        This is called by LiveKit's VoiceAssistant.
        """
        # Get the latest user message
        user_message = ""
        for msg in reversed(chat_ctx.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            yield llm.ChatChunk(
                choices=[llm.Choice(delta=llm.ChoiceDelta(content="I didn't catch that."))]
            )
            return
        
        # Process through workflow if available
        if self.current_workflow:
            timer_id = self.performance_monitor.start_timer("workflow_process") if self.performance_monitor else None
            
            try:
                # Invoke workflow
                result = await self.current_workflow.ainvoke({
                    "messages": [HumanMessage(content=user_message)],
                    "user_input": user_message
                })
                
                # Extract response
                response = ""
                if "response" in result:
                    response = result["response"]
                elif "messages" in result and result["messages"]:
                    last_msg = result["messages"][-1]
                    if hasattr(last_msg, 'content'):
                        response = last_msg.content
                
                if not response:
                    response = "I need a moment to process that."
                
                # Stream message events
                if self.message_streamer:
                    # Stream any tool calls
                    if "tool_calls" in result:
                        for tool_call in result["tool_calls"]:
                            self.message_streamer.stream_message({
                                "role": "tool_call",
                                "content": f"{tool_call.get('name', 'tool')}: {tool_call.get('args', {})}",
                                "timestamp": time.time()
                            })
                    
                    # Stream the response
                    self.message_streamer.stream_message({
                        "role": "assistant",
                        "content": response,
                        "timestamp": time.time()
                    })
                
                # Yield response chunks for LiveKit
                for chunk in response:
                    yield llm.ChatChunk(
                        choices=[llm.Choice(delta=llm.ChoiceDelta(content=chunk))]
                    )
                
                if self.performance_monitor and timer_id:
                    self.performance_monitor.end_timer(timer_id)
                
            except Exception as e:
                logger.error(f"Workflow processing error: {e}")
                yield llm.ChatChunk(
                    choices=[llm.Choice(delta=llm.ChoiceDelta(
                        content="I encountered an error. Let me try again."
                    ))]
                )
        else:
            # Fallback response when no workflow
            yield llm.ChatChunk(
                choices=[llm.Choice(delta=llm.ChoiceDelta(
                    content="I'm here to help. Please select a client first."
                ))]
            )
    
    def set_workflow(self, workflow: CompiledGraph):
        """Set the current workflow."""
        self.current_workflow = workflow


# Maintain backward compatibility
VoiceInteractionHandler = LiveKitVoiceInteractionHandler
