"""
Optimized Voice Chat Component - Enhanced UI with client data caching.
"""

import gradio as gr
import uuid
import logging
import os
from typing import Dict, List, Tuple, Generator, Any, Optional, Callable
from fastrtc import WebRTC, ReplyOnPause, AlgoOptions, SileroVadOptions, AdditionalOutputs

logger = logging.getLogger(__name__)


class VoiceChatInterface:
    """Gradio interface wrapper for VoiceInteractionHandler with client data caching."""
    
    def __init__(self, voice_handler=None, workflow=None, workflow_factory: Optional[Callable] = None, client_data_loader: Optional[Callable] = None):
        self.voice_handler = voice_handler
        self.workflow = workflow
        self.workflow_factory = workflow_factory
        self.client_data_loader = client_data_loader  # Function to load client data by user_id
    
    def _validate_history(self, history: List[Dict]) -> List[Dict]:
        """Validate chatbot history."""
        if not history:
            return []
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg
        ]
    
    def update_client_data(self, user_id: str) -> str:
        """Update client data in voice handler and return status."""
        chatbot, thread_id = self.start_new_conversation()

        if not user_id or not user_id.strip():
            return chatbot, thread_id, "❌ No client ID provided"
        
        try:
            if self.client_data_loader and self.voice_handler:
                # Load client data
                logger.info(f"Loading client data for user_id: {user_id}")
                client_data = self.client_data_loader(user_id)
                
                if client_data:
                    # Update voice handler with new client data
                    self.voice_handler.update_client_data(user_id, client_data)
                    
                    return chatbot, thread_id, f"✅ Client data loaded for ID: {user_id}"
                else:
                    return chatbot, thread_id, f"❌ No data found for client ID: {user_id}"
            else:
                return chatbot, thread_id, "❌ Voice handler or data loader not available"
                
        except Exception as e:
            logger.error(f"Error updating client data for {user_id}: {e}")
            return chatbot, thread_id, f"❌ Error loading client data: {str(e)}"
    
    def process_text_input(self, text_input: str, chatbot_history: List[Dict], thread_id: str) -> Generator:
        """Process text input with streaming."""
        if not text_input.strip():
            yield None, chatbot_history, ""
            return
            
        try:
            history = self._validate_history(chatbot_history)
            
            if self.voice_handler:
                # Use cached workflow from voice handler
                for result in self.voice_handler.process_text_input(
                    text_input.strip(), None, history, thread_id
                ):
                    audio, chatbot, _ = result if len(result) >= 3 else (*result, "")
                    yield audio, self._validate_history(chatbot) if chatbot else history, ""
            else:
                # Simple fallback response
                new_history = history + [
                    {"role": "user", "content": text_input},
                    {"role": "assistant", "content": "I'm here to help! How can I assist you today?"}
                ]
                yield None, new_history, ""
                    
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            yield None, history + [{"role": "assistant", "content": "I encountered an error processing your request."}], ""
    
    def process_audio_input(self, audio_input, chatbot_history: List[Dict], thread_id: str) -> Generator:
        """Process audio input."""
        if audio_input is None:
            yield audio_input
            return
            
        try:
            history = self._validate_history(chatbot_history)
            
            if self.voice_handler:
                # Use cached workflow from voice handler
                for result in self.voice_handler.process_audio_input(
                    audio_input, None, history, thread_id
                ):
                    yield result
            else:
                # Fallback for audio when handler not available
                yield AdditionalOutputs(history + [{"role": "assistant", "content": "Voice processing is not available."}])
                    
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            yield AdditionalOutputs(history + [{"role": "assistant", "content": "Error occurred during audio processing."}])
    
    def start_new_conversation(self) -> Tuple[List, str]:
        """Start new conversation."""
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
        except:
            pass
        new_id = str(uuid.uuid4())
        logger.info(f"New conversation: {new_id}")
        return [], new_id


def create_voice_chat_block(voice_handler=None, workflow=None, workflow_factory: Optional[Callable] = None, client_data_loader: Optional[Callable] = None, theme=None) -> gr.Blocks:
    """Create an optimized voice chat interface with client data caching."""
    
    interface = VoiceChatInterface(voice_handler, workflow, workflow_factory, client_data_loader)
    
    # Enhanced CSS for better styling
    voice_chat_css = """
    .voice-chat-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    .voice-header {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.2);
    }
    
    .voice-controls {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border: 1px solid #f1f5f9;
    }
    
    .conversation-area {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border: 1px solid #f1f5f9;
    }
    
    .input-controls {
        display: flex;
        gap: 8px;
        align-items: flex-end;
        margin-top: 12px;
    }
    
    .voice-status {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        text-align: center;
        margin-bottom: 12px;
    }
    
    .conversation-header {
        color: #475569;
        font-weight: 600;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .action-buttons {
        display: flex;
        gap: 8px;
        justify-content: space-between;
        margin-top: 12px;
    }
    
    .webrtc-container {
        border: 2px dashed #10b981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        transition: all 0.3s ease;
    }
    
    .webrtc-container:hover {
        border-color: #059669;
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    }
    
    .mic-icon {
        font-size: 24px;
        margin-bottom: 8px;
        display: block;
    }
    
    .client-status {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 12px;
        margin-bottom: 12px;
    }
    """
    
    with gr.Blocks(css=voice_chat_css) as block:
        voice_thread_id = gr.State(value=str(uuid.uuid4()))
        current_user_id = gr.State(value="")
        client_status = gr.State(value="❌ No client selected")

        with gr.Column(elem_classes="voice-chat-container"):
            # Header with client status
            gr.HTML("""
                <div class="voice-header">
                    <h3 style="margin: 0; font-size: 18px;">🎙️ Voice Agent</h3>
                </div>
            """)
            
            # Client status indicator
            client_status_display = gr.Markdown(value="❌ No client selected", elem_classes="client-status")
            
            # Voice Controls Section
            with gr.Column(elem_classes="voice-controls"):
                audio = WebRTC(
                    label="",
                    mode="send-receive",
                    modality="audio",
                    icon="asset/speaker-svgrepo-com.svg",
                    button_labels={"start": "🎙️ Start Speaking", "stop": "⏹️ Stop"},
                    track_constraints={
                    # Optimized for phone call quality and noise handling
                    "echoCancellation": {"exact": True},
                    "noiseSuppression": {"ideal": True},  # Ideal rather than exact for flexibility
                    "autoGainControl": {"exact": True},
                    "sampleRate": {"ideal": 1000},  # Phone quality sample rate
                    "sampleSize": {"ideal": 16},
                    "channelCount": {"exact": 1},
                    # Google-specific optimizations for call center use
                    "googEchoCancellation": {"exact": True},
                    "googNoiseSuppression": {"ideal": True},
                    "googAutoGainControl": {"exact": True},
                    "googHighpassFilter": {"exact": True},  # Remove low-frequency line noise
                    "googTypingNoiseDetection": {"ideal": True},
                    # Call center specific optimizations
                    "googBeamforming": {"ideal": True},  # Focus on primary speaker
                    "googExperimentalNoiseSuppression": {"ideal": True},
                },
                    elem_classes="webrtc-container"
                )
                
                with gr.Row(elem_classes="action-buttons"):
                    new_chat_btn = gr.Button(
                        "🗼 New Conversation", 
                        variant="primary", 
                        size="sm",
                        scale=1
                    )
                    refresh_client_btn = gr.Button(
                        "🔄 Refresh Client", 
                        variant="secondary", 
                        size="sm",
                        scale=1
                    )
            
            # Conversation Area
            with gr.Column(elem_classes="conversation-area"):
            
                if voice_handler:
                    text_chat_audio = gr.Audio(
                        label="", 
                        autoplay=True,
                        visible=False,
                        interactive=False,
                        streaming=True,
                        elem_id="response-audio"
                    )
                
                chatbot = gr.Chatbot(
                    label="",
                    type="messages",
                    value=[],
                    height="35vh",
                    bubble_full_width=False,
                    autoscroll=True,
                    render_markdown=True,
                    elem_classes="streaming-chatbot",
                    show_label=False,
                    container=False
                )
                
                # Enhanced input area
                with gr.Row(elem_classes="input-controls"):
                    text_input = gr.Textbox(
                        placeholder="💬 Type your message here...",
                        show_label=False,
                        container=False,
                        scale=4,
                        lines=1,
                        max_lines=3,
                        elem_id="text-input-field"
                    )
                    with gr.Column(scale=1):
                        send_btn = gr.Button(
                            "📤 Send", 
                            variant="primary", 
                            size="sm", 
                            scale=1,
                            min_width=80
                        )
                        stop_audio_btn = gr.Button(
                            "🔇 Stop", 
                            variant="secondary", 
                            size="sm",
                            scale=1,
                            min_width=80
                        )
        
        # Handlers
        def handle_additional_outputs(output):
            """Handle chatbot updates from audio."""
            try:
                if isinstance(output, list):
                    if (output and len(output) == 3 and output[0] == 'add' and 
                        isinstance(output[1], list) and isinstance(output[2], dict)):
                        return gr.update()
                    
                    return [
                        {"role": msg.get("role", ""), "content": msg.get("content", "")}
                        for msg in output
                        if isinstance(msg, dict) and msg.get('role') and msg.get('content')
                    ]
                
                elif hasattr(output, 'args') and output.args:
                    data = output.args[0]
                    if isinstance(data, list):
                        return [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in data
                            if isinstance(msg, dict) and 'role' in msg and 'content' in msg
                        ]
                
                return gr.update()
                
            except Exception as e:
                logger.error(f"Output handling error: {e}")
                return gr.update()
        
        # Connect events
        send_btn.click(
            fn=interface.process_text_input,
            inputs=[text_input, chatbot, voice_thread_id],
            outputs=[text_chat_audio, chatbot, text_input] if voice_handler else [chatbot, text_input],
            queue=True
        )
        
        text_input.submit(
            fn=interface.process_text_input,
            inputs=[text_input, chatbot, voice_thread_id],
            outputs=[text_chat_audio, chatbot, text_input] if voice_handler else [chatbot, text_input],
            queue=True
        )
        
        # Audio streaming (only if voice handler available)
        if voice_handler:
            audio.stream(
                ReplyOnPause(
                    interface.process_audio_input,
                    input_sample_rate=16000,
                    algo_options=AlgoOptions(
                        # Balanced for call center responsiveness vs accuracy
                        audio_chunk_duration=0.6,  # Responsive but not too quick
                        # Moderate threshold - catch genuine speech but avoid line noise
                        started_talking_threshold=0.4,  # Balanced for phone calls
                        # Lower threshold for ongoing speech detection (people talk continuously)
                        speech_threshold=0.3,  # Allow for natural pauses in speech
                    ),
                    model_options=SileroVadOptions(
                        # Optimized for phone call scenarios
                        threshold=0.7,  # Lower than noisy environment but higher than quiet
                        # Quick response for natural conversation flow
                        min_speech_duration_ms=200,  # Catch short responses like "yes", "no"
                        # Moderate silence duration - allow for thinking time but be responsive
                        min_silence_duration_ms=1000,  # Balance between responsiveness and accuracy
                        # Optimized window size for phone quality audio
                        window_size_samples=512,  # Good for 8kHz phone audio
                        # Minimal padding to avoid cutting off speech but stay responsive
                        speech_pad_ms=50,  # Just enough to capture word boundaries
                        # Additional phone-optimized settings
                        max_speech_duration_s=10.0,  # Prevent extremely long segments
                    ),
                    # can_interrupt=True,
                ),
                inputs=[audio, chatbot, voice_thread_id],
                outputs=[audio],
            )
            
            audio.on_additional_outputs(
                handle_additional_outputs,
                outputs=[chatbot],
                queue=False,
                show_progress="hidden"
            )
            
            stop_audio_btn.click(
                fn=lambda: None,
                outputs=[text_chat_audio],
                queue=False
            )
        
        new_chat_btn.click(
            fn=interface.start_new_conversation,
            outputs=[chatbot, voice_thread_id],
            queue=False
        )
        
        # Refresh client data button
        refresh_client_btn.click(
            fn=interface.update_client_data,
            inputs=[current_user_id],
            outputs=[chatbot, voice_thread_id, client_status_display],
            queue=False
        )
        
        # Store references for external access
        block.current_user_id = current_user_id
        block.client_status = client_status
        block.client_status_display = client_status_display
        block.interface = interface
        
    return block


def create_voice_chat_tab(voice_handler=None, workflow=None, workflow_factory: Optional[Callable] = None, client_data_loader: Optional[Callable] = None, tab_name: str = "🎙️ Voice Assistant") -> gr.Tab:
    """Create voice chat tab with enhanced styling and client data caching."""
    with gr.Tab(tab_name) as tab:
        voice_block = create_voice_chat_block(voice_handler, workflow, workflow_factory, client_data_loader)
        # Return both tab and block for state access
        tab.voice_block = voice_block
    return tab


# Export main functions
__all__ = [
    'VoiceChatInterface',
    'create_voice_chat_block',
    'create_voice_chat_tab',
]