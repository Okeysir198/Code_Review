"""
Production-ready Voice Interaction Handler for AI-powered conversational agents.
Enhanced with noise cancellation, connection stability, and comprehensive message streaming.
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
from fastrtc import AdditionalOutputs
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
    
    def get_avg_metric(self, metric_name: str) -> float:
        """Get average value for a metric."""
        values = self.metrics.get(metric_name, [])
        return sum(values) / len(values) if values else 0.0
    
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
    
    def __init__(self, max_handlers: int = 10):
        self.handlers = []
        self.max_handlers = max_handlers
        self.message_cache = set()
        self.cache_ttl = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def add_handler(self, handler: Callable[[str, str], None]):
        """Add a message handler with automatic cleanup."""
        if len(self.handlers) >= self.max_handlers:
            logger.warning(f"Too many handlers ({len(self.handlers)}), removing oldest")
            self.handlers.pop(0)
        self.handlers.append(handler)
    
    def stream_message(self, role: str, content: str, force: bool = False):
        """Stream message with deduplication."""
        if not force:
            # Clean old cache entries periodically
            current_time = time.time()
            if current_time - self.last_cleanup > 60:  # Cleanup every minute
                self.message_cache.clear()
                self.last_cleanup = current_time
            
            # Create unique message hash
            msg_hash = hashlib.md5(f"{role}:{content}".encode()).hexdigest()
            if msg_hash in self.message_cache:
                return
            self.message_cache.add(msg_hash)
        
        # Stream to all handlers
        for handler in self.handlers[:]:  # Use slice to avoid issues if handler list changes
            try:
                handler(role, content)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
                # Remove failed handler
                if handler in self.handlers:
                    self.handlers.remove(handler)
    
    def clear_handlers(self):
        """Clear all handlers."""
        self.handlers.clear()
        self.message_cache.clear()


class ComponentHealthMonitor:
    """Monitor health of system components."""
    
    def __init__(self):
        self.component_health = {
            'stt': True,
            'tts': True,
            'workflow': True
        }
        self.failure_counts = defaultdict(int)
        self.last_failure_time = defaultdict(float)
        self.recovery_threshold = 3  # attempts before marking as failed
        self.recovery_window = 300  # 5 minutes
    
    def check_component_health(self, component: str, operation: Callable, *args, **kwargs):
        """Execute operation with health monitoring."""
        try:
            result = operation(*args, **kwargs)
            
            # Reset failure count on success
            if self.failure_counts[component] > 0:
                logger.info(f"{component} component recovered")
                self.failure_counts[component] = 0
                self.component_health[component] = True
            
            return result
            
        except Exception as e:
            current_time = time.time()
            
            # Reset failure count if outside recovery window
            if current_time - self.last_failure_time[component] > self.recovery_window:
                self.failure_counts[component] = 0
            
            self.failure_counts[component] += 1
            self.last_failure_time[component] = current_time
            
            # Mark as failed if too many failures
            if self.failure_counts[component] >= self.recovery_threshold:
                self.component_health[component] = False
                logger.error(f"{component} component marked as failed after {self.failure_counts[component]} failures")
            
            raise
    
    def is_healthy(self, component: str) -> bool:
        """Check if component is healthy."""
        return self.component_health.get(component, False)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all components."""
        return {
            'health': self.component_health.copy(),
            'failure_counts': dict(self.failure_counts),
            'last_failures': dict(self.last_failure_time)
        }


class VoiceInteractionHandler:
    """
    Production-ready voice interaction handler with enhanced stability,
    noise cancellation, and comprehensive message streaming.
    """
    
    def __init__(self, config: Dict[str, Any], workflow_factory: Optional[Callable] = None):
        # Core configuration
        self.config = self._setup_config(config or {})
        self.workflow_factory = workflow_factory
        
        # Enhanced components
        self.message_streamer = MessageStreamer()
        self.performance_monitor = PerformanceMonitor()
        self.health_monitor = ComponentHealthMonitor()
        
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
        self.processing_lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None
        self.concurrent_limit = 3
        self.active_requests = 0
        
        # Fallback responses for error scenarios
        self.fallback_responses = [
            "I'm having trouble processing that. Could you please repeat?",
            "Sorry, there seems to be a technical issue. Let me try again.",
            "I need a moment to process. Please hold on.",
            "Could you please rephrase that? I didn't catch it clearly."
        ]
        
        # Initialize models with health monitoring
        self.stt_model = self._init_stt() if self.config['configurable'].get('enable_stt_model') else None
        self.tts_model = self._init_tts() if self.config['configurable'].get('enable_tts_model') else None
        
        # Enhanced audio constraints for production WebRTC
        self.enhanced_audio_constraints = self._create_audio_constraints()
        
        logger.info("VoiceInteractionHandler initialized successfully")

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
        config.setdefault('audio', {
            'noise_suppression': {'enabled': True},
            'echo_cancellation': {'enabled': True},
            'auto_gain_control': {'enabled': True}
        })
        config.setdefault('performance', {
            'request_timeout': 30,
            'max_audio_duration': 30,
            'cache_ttl': 3600
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
        
        if self.config['logging'].get('console_output', True) and not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    def _create_audio_constraints(self) -> Dict[str, Any]:
        """Create production-grade audio constraints."""
        audio_config = self.config.get('audio', {})
        
        return {
            # Basic WebRTC constraints
            "echoCancellation": {"exact": audio_config.get('echo_cancellation', {}).get('enabled', True)},
            "noiseSuppression": {"exact": audio_config.get('noise_suppression', {}).get('enabled', True)},
            "autoGainControl": {"exact": audio_config.get('auto_gain_control', {}).get('enabled', True)},
            "sampleRate": {"ideal": 16000},
            "channelCount": {"exact": 1},
            
            # Advanced Google WebRTC constraints for maximum noise suppression
            "googNoiseSuppression": {"exact": True},
            "googNoiseSuppression2": {"exact": True},
            "googEchoCancellation": {"exact": True},
            "googEchoCancellation2": {"exact": True},
            "googAutoGainControl": {"exact": True},
            "googAutoGainControl2": {"exact": True},
            "googHighpassFilter": {"exact": True},
            "googTypingNoiseDetection": {"exact": True},
            "googAudioMirroring": {"exact": False},
            "googNoiseReduction": {"exact": True},
            
            # Additional stability constraints
            "googDAEchoCancellation": {"exact": True},
            "googExperimentalEchoCancellation": {"exact": True},
            "googExperimentalNoiseSuppression": {"exact": True},
            "googExperimentalAutoGainControl": {"exact": True}
        }

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
            self.health_monitor.component_health['stt'] = False
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
            self.health_monitor.component_health['tts'] = False
            raise

    def _validate_audio_input(self, audio_input: Tuple[int, np.ndarray]) -> bool:
        """Validate audio input for security and sanity."""
        try:
            if not isinstance(audio_input, tuple) or len(audio_input) != 2:
                logger.warning(f"Invalid audio format: {type(audio_input)}")
                return False
            
            sample_rate, audio_array = audio_input
            
            # Validate sample rate
            if not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 48000:
                logger.warning(f"Invalid sample rate: {sample_rate}")
                return False
            
            # Validate audio array
            if not isinstance(audio_array, np.ndarray):
                logger.warning(f"Audio data is not numpy array: {type(audio_array)}")
                return False
            
            # Check audio size (prevent memory attacks)
            max_duration = self.config['performance'].get('max_audio_duration', 30)
            max_samples = sample_rate * max_duration
            if audio_array.size > max_samples:
                logger.warning(f"Audio too long: {audio_array.size} samples (max: {max_samples})")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Audio validation error: {e}")
            return False

    def _normalize_audio(self, audio_chunk: tuple) -> tuple:
        """Enhanced audio normalization with validation and error handling."""
        try:
            if not self._validate_audio_input(audio_chunk):
                return (16000, np.array([0], dtype=np.int16))
            
            sample_rate, audio_array = audio_chunk
            
            # Handle empty arrays
            if audio_array.size == 0:
                return (sample_rate, np.array([0], dtype=np.int16))
            
            # Convert to int16 if needed
            if audio_array.dtype == np.float32:
                # Ensure values are in [-1, 1] range before conversion
                audio_array = np.clip(audio_array, -1.0, 1.0)
                audio_array = (audio_array * 32767).astype(np.int16)
            elif audio_array.dtype != np.int16:
                # Convert other types safely
                audio_array = audio_array.astype(np.float32)
                if np.max(np.abs(audio_array)) > 0:
                    audio_array = audio_array / np.max(np.abs(audio_array))
                audio_array = np.clip(audio_array, -1.0, 1.0)
                audio_array = (audio_array * 32767).astype(np.int16)
            
            return (sample_rate, audio_array)
            
        except Exception as e:
            logger.error(f"Audio normalization error: {e}")
            return (16000, np.array([0], dtype=np.int16))

    def _get_request_hash(self, audio_input: Tuple[int, np.ndarray]) -> str:
        """Generate hash for audio input to detect duplicates."""
        try:
            sample_rate, audio_array = audio_input
            # Hash based on audio characteristics, not raw data
            audio_stats = f"{sample_rate}_{audio_array.shape}_{np.mean(audio_array):.3f}_{np.std(audio_array):.3f}"
            return hashlib.md5(audio_stats.encode()).hexdigest()
        except Exception:
            return str(uuid.uuid4())

    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        
        # Clean request cache
        expired_keys = [
            key for key, (_, timestamp) in self.request_cache.items()
            if current_time - timestamp > 300  # 5 minutes
        ]
        for key in expired_keys:
            del self.request_cache[key]
        
        # Clean client data cache if expired
        if current_time - self.cache_timestamp > self.cache_ttl:
            self.cached_client_data = None
            self.cached_workflow = None
            self.cache_timestamp = 0

    def _print(self, label: str, text: str, color: str):
        """Print formatted message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"{Colors.BOLD}[{timestamp}] {color}[{label}]:{Colors.RESET} {color}{text}{Colors.RESET}")

    def _print_tool_response(self, tool_name: str, content: Any):
        """Print formatted tool response."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"{Colors.BOLD}[{timestamp}] {Colors.MAGENTA}[Tool Response - {tool_name}]:{Colors.RESET}")
        try:
            if isinstance(content, str):
                try:
                    formatted = json.dumps(json.loads(content), indent=2)
                except json.JSONDecodeError:
                    formatted = content
            elif isinstance(content, dict):
                formatted = json.dumps(content, indent=2)
            else:
                formatted = str(content)
        except Exception:
            formatted = str(content)
        print(f"{Colors.MAGENTA}{formatted}{Colors.RESET}")

    def _process_tool_messages(self, message, tracked_calls: Set[str]) -> List[Dict[str, str]]:
        """Process tool messages and return structured message data."""
        messages = []
        
        # Handle tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_id = getattr(tool_call, "id", str(id(tool_call)))
                tool_name = getattr(tool_call, "name", "unknown")
                tool_args = getattr(tool_call, "args", {})
                call_id = f"{tool_name}:{tool_id}"
                
                if call_id not in tracked_calls:
                    tool_msg = f"{tool_name}({tool_args})"
                    self._print("Tool Call", tool_msg, Colors.BLUE)
                    
                    # Add to messages list
                    messages.append({
                        "role": "tool_call",
                        "content": f"Calling {tool_name} with parameters: {tool_args}",
                        "tool_name": tool_name,
                        "tool_args": tool_args
                    })
                    
                    # Stream the message
                    self.message_streamer.stream_message("tool", f"CALL: {tool_msg}")
                    tracked_calls.add(call_id)
        
        # Handle tool responses
        if isinstance(message, ToolMessage):
            tool_name = getattr(message, "name", "Unknown Tool")
            content = getattr(message, "content", None)
            
            if content:
                self._print_tool_response(tool_name, content)
                
                # Add to messages list
                tool_response_content = content
                if isinstance(content, str) and len(content) > 200:
                    tool_response_content = content[:200] + "..."
                
                messages.append({
                    "role": "tool_response",
                    "content": f"Tool {tool_name} responded: {tool_response_content}",
                    "tool_name": tool_name,
                    "full_content": content
                })
                
                # Stream the response (only once per message)
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
        
        return messages

    def _extract_ai_response(self, workflow_result: Dict[str, Any]) -> str:
        """Extract AI response from workflow result."""
        if "error" in workflow_result:
            return workflow_result["error"]
        
        messages = workflow_result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
                return msg.content
        
        return "I apologize, but I couldn't generate a proper response. Please try again."

    def _format_chatbot(self, chatbot: List[Dict]) -> List[Dict]:
        """Format chatbot messages with validation."""
        formatted = []
        for msg in chatbot:
            if isinstance(msg, dict) and msg.get('role') and msg.get('content'):
                formatted.append({
                    "role": str(msg.get("role", "")),
                    "content": str(msg.get("content", ""))
                })
        return formatted

    # Public API Methods

    def add_message_handler(self, handler: Callable[[str, str], None]):
        """Add external message handler for real-time streaming."""
        self.message_streamer.add_handler(handler)

    def update_client_data(self, user_id: str, client_data: Dict[str, Any]):
        """Update cached client data and workflow with enhanced error handling."""
        try:
            current_time = time.time()
            
            if user_id != self.cached_user_id or not self.cached_workflow or current_time - self.cache_timestamp > self.cache_ttl:
                logger.info(f"Updating client data cache for user_id: {user_id}")
                
                self.cached_user_id = user_id
                self.cached_client_data = client_data
                self.cache_timestamp = current_time
                
                # Create new workflow with cached data
                if self.workflow_factory and client_data:
                    timer_id = self.performance_monitor.start_timer("workflow_creation")
                    self.cached_workflow = self.health_monitor.check_component_health(
                        'workflow', self.workflow_factory, client_data
                    )
                    self.performance_monitor.end_timer(timer_id)
                    logger.info(f"Created new workflow for user_id: {user_id}")
                else:
                    self.cached_workflow = None
                    logger.warning(f"No workflow factory or client data for user_id: {user_id}")
                    
        except Exception as e:
            logger.error(f"Error updating client data for user_id {user_id}: {e}")
            self.cached_workflow = None
            self.health_monitor.component_health['workflow'] = False

    def get_current_workflow(self) -> Optional[CompiledGraph]:
        """Get the current cached workflow."""
        return self.cached_workflow

    def optimize_audio_constraints(self, base_constraints: dict) -> dict:
        """Optimize audio constraints based on system performance."""
        optimized = base_constraints.copy()
        
        # Adaptive sample rate based on system load
        if self.active_requests > 1:
            optimized["sampleRate"] = {"ideal": 8000}  # Lower quality when busy
        else:
            optimized["sampleRate"] = {"ideal": 16000}  # High quality when available
        
        # Adjust processing settings based on component health
        if not self.health_monitor.is_healthy('stt'):
            optimized["noiseSuppression"] = {"exact": False}  # Reduce processing load
        
        return optimized

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'performance': self.performance_monitor.get_metrics_summary(),
            'health': self.health_monitor.get_health_summary(),
            'cache_stats': {
                'client_cache_age': time.time() - self.cache_timestamp if self.cache_timestamp else 0,
                'request_cache_size': len(self.request_cache),
                'active_requests': self.active_requests
            }
        }

    def process_message(self, user_message: str, workflow: CompiledGraph = None) -> Dict[str, Any]:
        """
        Process message through workflow with comprehensive message tracking.
        Returns all message types with metadata.
        """
        timer_id = self.performance_monitor.start_timer("process_message")
        
        try:
            # Use cached workflow if no workflow provided
            if workflow is None:
                workflow = self.get_current_workflow()
            
            if not workflow:
                error_msg = "No workflow available. Please select a client first."
                return {
                    "messages": [{"role": "error", "content": error_msg}],
                    "error": error_msg,
                    "all_messages": [{"role": "error", "content": error_msg}]
                }
            
            workflow_input = {"messages": [HumanMessage(content=user_message)]}
            config = {"configurable": self.config.get('configurable', {})}
            
            tracked_calls = set()
            tracked_messages = set()
            all_messages = []  # Store ALL messages with metadata
            final_result = None
            
            # Add user message
            user_msg = {
                "role": "user",
                "content": user_message,
                "timestamp": time.time()
            }
            all_messages.append(user_msg)
            
            # Stream user message only once
            user_msg_id = f"user_{hash(user_message)}"
            if user_msg_id not in tracked_messages:
                self.message_streamer.stream_message("user", user_message)
                tracked_messages.add(user_msg_id)
            
            # Process workflow events
            for event in workflow.stream(workflow_input, config=config, stream_mode="values"):
                if event and isinstance(event, dict) and "messages" in event:
                    final_result = event
                    
                    if event["messages"]:
                        latest_message = event["messages"][-1]
                        
                        # Process tool messages and add to all_messages
                        tool_messages = self._process_tool_messages(latest_message, tracked_calls)
                        for tool_msg in tool_messages:
                            tool_msg["timestamp"] = time.time()
                            all_messages.append(tool_msg)
                        
                        # Process AI messages
                        if isinstance(latest_message, AIMessage) and latest_message.content:
                            ai_msg = {
                                "role": "ai",
                                "content": latest_message.content,
                                "timestamp": time.time(),
                                "is_final_response": True  # Mark as TTS-eligible
                            }
                            all_messages.append(ai_msg)
                            
                            # Stream AI message only once
                            ai_msg_id = f"ai_{id(latest_message)}"
                            if ai_msg_id not in tracked_messages:
                                self.message_streamer.stream_message("ai", latest_message.content)
                                tracked_messages.add(ai_msg_id)
            
            # Prepare final result with all message types
            if final_result:
                final_result["all_messages"] = all_messages
            else:
                final_result = {
                    "messages": [],
                    "error": "No valid events received",
                    "all_messages": all_messages
                }
            
            self.performance_monitor.end_timer(timer_id)
            return final_result
            
        except Exception as e:
            self.performance_monitor.end_timer(timer_id)
            logger.error(f"Workflow error: {e}")
            error_msg = f"Workflow execution error: {str(e)}"
            self.message_streamer.stream_message("error", error_msg)
            
            return {
                "messages": [],
                "error": error_msg,
                "all_messages": [{"role": "error", "content": error_msg, "timestamp": time.time()}]
            }

    def process_text_input(self, text_input: str, workflow: CompiledGraph = None,
                          gradio_chatbot: Optional[List[Dict[str, str]]] = None,
                          thread_id: Optional[Union[str, int]] = None) -> Generator:
        """Process text input with comprehensive message streaming and TTS for AI responses only."""
        try:
            chatbot = gradio_chatbot or []
            timer_id = self.performance_monitor.start_timer("process_text_input")
            
            # Set thread ID
            self.config['configurable']["thread_id"] = str(thread_id or uuid.uuid4())
            
            # Add user message to chatbot
            self._print("Text Input", text_input, Colors.YELLOW)
            chatbot.append({"role": "user", "content": text_input})
            yield None, chatbot, ""

            # Process and get response with all message types
            workflow_result = self.process_message(text_input, workflow)
            all_messages = workflow_result.get("all_messages", [])
            
            # Add ALL message types to chatbot for display
            for msg in all_messages:
                if msg["role"] in ["user"]:  # Skip user since already added
                    continue
                chatbot.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            yield None, chatbot, ""
            
            # Generate TTS ONLY for AI messages marked as final responses
            if self.tts_model and self.health_monitor.is_healthy('tts') and self.config['configurable'].get('enable_tts_model'):
                ai_responses = [msg for msg in all_messages if msg["role"] == "ai" and msg.get("is_final_response", False)]
                
                for ai_msg in ai_responses:
                    response_text = ai_msg["content"].strip()
                    if response_text:
                        self._print("AI Response", response_text, Colors.GREEN)
                        try:
                            for audio_chunk in self.health_monitor.check_component_health(
                                'tts', self.tts_model.stream_text_to_speech, response_text
                            ):
                                yield self._normalize_audio(audio_chunk), chatbot, ""
                        except Exception as e:
                            logger.error(f"TTS generation failed: {e}")
                            # Continue without TTS
                            break
            
            self.performance_monitor.end_timer(timer_id)
                
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            chatbot.append({"role": "assistant", "content": "Sorry, an error occurred while processing your request."})
            yield None, chatbot, ""

    def process_audio_input(self, audio_input: Tuple[int, np.ndarray], workflow: CompiledGraph = None,
                           gradio_chatbot: Optional[List[Dict[str, str]]] = None,
                           thread_id: Optional[Union[str, int]] = None) -> Generator:
        """Process audio input with enhanced stability and comprehensive message streaming."""
        timer_id = self.performance_monitor.start_timer("process_audio_input")
        
        try:
            # Rate limiting check
            if self.active_requests >= self.concurrent_limit:
                logger.warning("Rate limit exceeded, dropping audio request")
                yield (audio_input[0], np.array([0], dtype=np.int16))
                yield AdditionalOutputs(self._format_chatbot(gradio_chatbot or []))
                return
            
            self.active_requests += 1
            chatbot = gradio_chatbot or []
            sample_rate = audio_input[0]
            
            # Validate audio input
            if not self._validate_audio_input(audio_input):
                logger.warning("Invalid audio input received")
                yield (sample_rate, np.array([0], dtype=np.int16))
                yield AdditionalOutputs(self._format_chatbot(chatbot))
                return
            
            # Check if audio is valid and models are available
            if (not audio_input[1].size or 
                not self.stt_model or 
                not self.health_monitor.is_healthy('stt') or
                not self.config['configurable'].get('enable_stt_model')):
                
                yield (sample_rate, np.array([0], dtype=np.int16))
                yield AdditionalOutputs(self._format_chatbot(chatbot))
                return
            
            # Check for duplicate requests
            self._cleanup_cache()
            request_hash = self._get_request_hash(audio_input)
            current_time = time.time()
            
            if request_hash in self.request_cache:
                cached_result, timestamp = self.request_cache[request_hash]
                if current_time - timestamp < 30:  # 30 second dedup window
                    logger.info("Using cached response for similar audio")
                    for result in cached_result:
                        yield result
                    return
            
            # Transcribe audio with health monitoring
            stt_timer = self.performance_monitor.start_timer("stt_transcription")
            try:
                stt_result = self.health_monitor.check_component_health(
                    'stt', self.stt_model.transcribe, audio_input
                )
                stt_text = (stt_result.get("text", "") if isinstance(stt_result, dict) 
                           else str(stt_result)).strip()
                self.performance_monitor.end_timer(stt_timer)
            except Exception as e:
                self.performance_monitor.end_timer(stt_timer)
                logger.error(f"STT transcription failed: {e}")
                stt_text = ""
            
            if not stt_text:
                yield (sample_rate, np.array([0], dtype=np.int16))
                yield AdditionalOutputs(self._format_chatbot(chatbot))
                return
            
            # Process conversation with workflow
            self._print("Voice Input", stt_text, Colors.YELLOW)
            chatbot.append({"role": "user", "content": stt_text})
            
            # Set thread ID and get response with all message types
            self.config['configurable']["thread_id"] = str(thread_id or uuid.uuid4())
            
            workflow_timer = self.performance_monitor.start_timer("workflow_processing")
            try:
                workflow_result = self.process_message(stt_text, workflow)
                all_messages = workflow_result.get("all_messages", [])
                self.performance_monitor.end_timer(workflow_timer)
            except Exception as e:
                self.performance_monitor.end_timer(workflow_timer)
                logger.error(f"Workflow processing failed: {e}")
                # Use fallback response
                fallback_response = random.choice(self.fallback_responses)
                chatbot.append({"role": "assistant", "content": fallback_response})
                yield AdditionalOutputs(self._format_chatbot(chatbot))
                yield (sample_rate, np.array([0], dtype=np.int16))
                return
            
            # Add ALL message types to chatbot for display (except user, already added)
            for msg in all_messages:
                if msg["role"] in ["user"]:  # Skip user since already added
                    continue
                    
                # Format different message types appropriately
                if msg["role"] == "tool_call":
                    display_content = f"🔧 {msg['content']}"
                elif msg["role"] == "tool_response":
                    display_content = f"📋 {msg['content']}"
                elif msg["role"] == "ai":
                    display_content = msg["content"]
                elif msg["role"] == "error":
                    display_content = f"❌ {msg['content']}"
                else:
                    display_content = msg["content"]
                
                chatbot.append({
                    "role": msg["role"],
                    "content": display_content
                })
            
            # Update UI with final chatbot state
            yield AdditionalOutputs(self._format_chatbot(chatbot))
            
            # Generate TTS ONLY for AI messages marked as final responses
            results = []
            if (self.tts_model and 
                self.health_monitor.is_healthy('tts') and 
                self.config['configurable'].get('enable_tts_model')):
                
                ai_responses = [msg for msg in all_messages if msg["role"] == "ai" and msg.get("is_final_response", False)]
                
                for ai_msg in ai_responses:
                    response_text = ai_msg["content"].strip()
                    if response_text:
                        self._print("AI Response", response_text, Colors.GREEN)
                        
                        tts_timer = self.performance_monitor.start_timer("tts_generation")
                        try:
                            for chunk in self.health_monitor.check_component_health(
                                'tts', self.tts_model.stream_text_to_speech, response_text
                            ):
                                normalized_chunk = self._normalize_audio(chunk)
                                results.append(normalized_chunk)
                                yield normalized_chunk
                            self.performance_monitor.end_timer(tts_timer)
                        except Exception as e:
                            self.performance_monitor.end_timer(tts_timer)
                            logger.error(f"TTS generation failed: {e}")
                            # Fallback to silence
                            fallback_audio = (sample_rate, np.array([0], dtype=np.int16))
                            results.append(fallback_audio)
                            yield fallback_audio
                            break
            else:
                # No TTS available or disabled
                fallback_audio = (sample_rate, np.array([0], dtype=np.int16))
                results.append(fallback_audio)
                yield fallback_audio
            
            # Cache successful results
            if results:
                self.request_cache[request_hash] = (results, current_time)
            
            self.performance_monitor.end_timer(timer_id)
            
        except Exception as e:
            logger.error(f"Critical audio processing error: {e}")
            chatbot.append({"role": "assistant", "content": "Sorry, a critical error occurred while processing your voice input."})
            yield (audio_input[0] if audio_input else 16000, np.array([0], dtype=np.int16))
            yield AdditionalOutputs(self._format_chatbot(chatbot))
        finally:
            if hasattr(self, 'active_requests'):
                self.active_requests = max(0, self.active_requests - 1)

    def set_log_level(self, level: str):
        """Set logging level at runtime."""
        self.config['logging']['level'] = level
        self._setup_logging()
        logger.info(f"Log level changed to: {level}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for monitoring."""
        return {
            "timestamp": time.time(),
            "component_health": self.health_monitor.get_health_summary(),
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
            "audio_constraints": self.enhanced_audio_constraints
        }

    def cleanup_resources(self):
        """Clean up resources and connections."""
        try:
            logger.info("Cleaning up VoiceInteractionHandler resources...")
            
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

    def __del__(self):
        """Destructor to ensure clean shutdown."""
        try:
            self.cleanup_resources()
        except Exception:
            pass  # Ignore errors during destruction