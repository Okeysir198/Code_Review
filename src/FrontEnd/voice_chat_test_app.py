# src/FrontEnd/voice_chat_test_app.py
"""
Voice Chat Test App with FastRTC Integration and Terminal Console
3 Columns: Client Info | Mandate History | Console Logs with Real-time Updates
"""

import gradio as gr
import sys
import logging
import io
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Generator, Callable, Union, Set
import numpy as np

sys.path.append("../..")

from fastrtc import WebRTC, ReplyOnPause, AlgoOptions, SileroVadOptions, AdditionalOutputs
from src.Agents.call_center_agent.data.client_data_fetcher import get_client_data
from src.VoiceHandler import VoiceInteractionHandler
from src.Agents.graph_call_center_agent import create_call_center_agent
from src.Database.CartrackSQLDatabase import get_client_debit_mandates
from langchain_ollama import ChatOllama
from app_config import CONFIG


class ConsoleCapture:
    """Captures console output and conversation logs separately with deduplication."""
    
    def __init__(self):
        self.output = ""
        self.conversation_output = ""
        self.buffer = io.StringIO()
        self.last_messages = set()  # Track recent messages to prevent duplication
        self.max_tracked = 100  # Limit tracked messages to prevent memory issues
    
    def write(self, text):
        self.buffer.write(text)
        self.output += text
        return len(text)
    
    def log_conversation(self, role: str, content: str):
        """Log conversation messages with timestamp and deduplication."""
        # Create message hash for deduplication
        msg_hash = hash(f"{role}:{content}")
        
        # Skip if we've seen this exact message recently
        if msg_hash in self.last_messages:
            return
        
        # Add to tracking and limit size
        self.last_messages.add(msg_hash)
        if len(self.last_messages) > self.max_tracked:
            # Remove oldest half of tracked messages
            self.last_messages = set(list(self.last_messages)[self.max_tracked//2:])
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        if role.lower() in ["user", "human"]:
            conv_msg = f"[{timestamp}] 👤 USER: {content}\n"
        elif role.lower() in ["assistant", "ai"]:
            conv_msg = f"[{timestamp}] 🤖 AI: {content}\n"
        elif role.lower() == "tool":
            conv_msg = f"[{timestamp}] 🔧 TOOL: {content}\n"
        elif role.lower() == "system":
            conv_msg = f"[{timestamp}] 📍 SYSTEM: {content}\n"
        elif role.lower() == "error":
            conv_msg = f"[{timestamp}] ❌ ERROR: {content}\n"
        else:
            conv_msg = f"[{timestamp}] {role.upper()}: {content}\n"
        
        self.conversation_output += conv_msg
        self.write(conv_msg)
    
    def flush(self):
        pass
    
    def get_output(self):
        return self.output
    
    def get_conversation_output(self):
        return self.conversation_output
    
    def clear(self):
        self.output = ""
        self.conversation_output = ""
        self.buffer = io.StringIO()
        self.last_messages.clear()


# Global console capture
console_capture = ConsoleCapture()


class ConsoleHandler(logging.Handler):
    """Custom logging handler for console capture."""
    
    def __init__(self, capture):
        super().__init__()
        self.capture = capture
    
    def emit(self, record):
        msg = self.format(record)
        self.capture.write(f"{msg}\n")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = ConsoleHandler(console_capture)
logging.getLogger().addHandler(console_handler)


class VoiceTestInterface:
    """Voice test interface with FastRTC integration and message streaming."""
    
    def __init__(self):
        self.voice_handler = None
        self.current_client_data = None
        self.current_user_id = ""
    
    def update_client_data(self, user_id: str) -> str:
        """Update client data and setup voice handler with message streaming (prevent duplicate handlers)."""
        try:
            if not user_id or not user_id.strip():
                return "❌ No client ID provided"
            
            console_capture.write(f"\n🔄 UPDATING CLIENT DATA: {user_id}\n")
            
            # Load client data
            client_data = get_client_data(user_id)
            if not client_data:
                return f"❌ No data found for client ID: {user_id}"
            
            self.current_client_data = client_data
            self.current_user_id = user_id
            
            # Create workflow factory
            def workflow_factory(client_data_param):
                return self.create_voice_workflow(client_data_param)
            
            # Create voice handler with message streaming (only once)
            self.voice_handler = VoiceInteractionHandler(CONFIG, workflow_factory)
            
            # Clear any existing handlers and add our handler only once
            self.voice_handler.message_streamer.handlers.clear()
            self.voice_handler.add_message_handler(console_capture.log_conversation)
            
            # Update client data
            self.voice_handler.update_client_data(user_id, client_data)
            
            client_name = client_data.get('profile', {}).get('client_info', {}).get('client_full_name', 'Unknown')
            console_capture.write(f"✅ Voice handler ready for: {client_name}\n")
            
            return f"✅ Ready for client: {client_name}"
            
        except Exception as e:
            console_capture.write(f"❌ Error updating client data: {e}\n")
            return f"❌ Error: {str(e)}"
    
    def create_voice_workflow(self, client_data: Dict[str, Any]) -> Optional[Any]:
        """Create call center agent workflow."""
        try:
            if client_data:
                console_capture.write(f"🔧 Creating workflow with memory: {CONFIG.get('configurable', {}).get('use_memory', False)}...\n")
                
                llm = ChatOllama(model="qwen2.5:14b-instruct", temperature=0, num_ctx=32000)
                
                workflow = create_call_center_agent(
                    model=llm,
                    client_data=client_data,
                    config=CONFIG,
                    verbose=True
                )
                
                console_capture.write("✅ Workflow created successfully\n")
                return workflow
            return None
        except Exception as e:
            console_capture.write(f"❌ Error creating workflow: {e}\n")
            return None
    
    def process_audio_input(self, audio_input, chatbot_history: List[Dict], thread_id: str) -> Generator:
        """Process audio input with memory persistence."""
        if audio_input is None:
            yield audio_input
            return
            
        try:
            console_capture.write(f"\n{'='*60}\n")
            console_capture.write(f"🎤 PROCESSING AUDIO INPUT\n")
            console_capture.write(f"🧵 Thread ID: {thread_id}\n")
            console_capture.write(f"🕒 Time: {datetime.now().strftime('%H:%M:%S')}\n")
            console_capture.write(f"{'='*60}\n")
            
            if self.voice_handler:
                for result in self.voice_handler.process_audio_input(
                    audio_input, None, chatbot_history, thread_id
                ):
                    console_capture.write(f"🔊 Audio processing result received\n")
                    yield result
            else:
                console_capture.write("❌ Voice handler not available\n")
                yield AdditionalOutputs(chatbot_history + [{"role": "assistant", "content": "Voice processing not available"}])
                    
        except Exception as e:
            console_capture.write(f"❌ Audio processing error: {e}\n")
            yield AdditionalOutputs(chatbot_history + [{"role": "assistant", "content": f"Audio error: {str(e)}"}])


def get_client_info_display(user_id: str) -> str:
    """Get formatted client info for verification."""
    try:
        if not user_id or not user_id.strip():
            return "## 👤 **Client Information**\n\nEnter a client ID to load information for verification."
        
        client_data = get_client_data(user_id)
        if not client_data:
            return f"## ❌ **Client Not Found**\n\nNo data available for client ID: {user_id}"
        
        # Extract key info
        profile = client_data.get("profile", {})
        client_info = profile.get("client_info", {})
        vehicles = profile.get("vehicles", [])
        account_aging = client_data.get("account_aging", {})
        
        client_name = client_info.get("client_full_name", "Unknown")
        title = client_info.get("title", "Mr/Ms")
        id_number = client_info.get("id_number", "N/A")
        email = client_info.get("email_address", "N/A")
        username = profile.get("user_name", "N/A")
        
        # Calculate outstanding
        try:
            total = float(account_aging.get("xbalance", 0))
            current = float(account_aging.get("x0", 0))
            outstanding = max(total - current, 0.0)
            outstanding_str = f"R {outstanding:.2f}"
        except:
            outstanding_str = "R 0.00"
        
        # Vehicle info
        vehicle_info = "🚫 No vehicles"
        if vehicles and len(vehicles) > 0:
            vehicle = vehicles[0]
            vehicle_info = f"🚗 {vehicle.get('registration', 'N/A')} - {vehicle.get('make', 'N/A')} {vehicle.get('model', 'N/A')} ({vehicle.get('color', 'N/A')})"
        
        return f"""## 👤 **{title} {client_name}**

### 🔍 **Verification Details**
| Field | Value |
|:------|:------|
| 🆔 **User ID** | `{user_id}` |
| 📄 **ID Number** | `{id_number}` |
| 📧 **Email** | `{email}` |
| 👤 **Username** | `{username}` |
| 💰 **Outstanding** | **`{outstanding_str}`** |
| 📊 **Total Balance** | `R {account_aging.get('xbalance', '0.00')}` |

### 🚗 **Vehicle Information**
{vehicle_info}

### 📅 **Account Aging Breakdown**
| Period | Amount |
|:-------|:-------|
| 🟢 **Current** | `R {account_aging.get('x0', '0.00')}` |
| 🟡 **30 Days** | `R {account_aging.get('x30', '0.00')}` |
| 🟠 **60 Days** | `R {account_aging.get('x60', '0.00')}` |
| 🔴 **90 Days** | `R {account_aging.get('x90', '0.00')}` |
| ⚠️ **120+ Days** | `R {account_aging.get('x120', '0.00')}` |

---
*Use this information to verify client identity during the call*
"""
    
    except Exception as e:
        return f"## ❌ **Error Loading Client Data**\n\nError: {str(e)}"


def get_mandate_history(user_id: str) -> str:
    """Get formatted mandate history (last 5 records)."""
    try:
        if not user_id or not user_id.strip():
            return "## 🏦 **Mandate History**\n\nNo mandate history available - load a client first."
        
        mandates = get_client_debit_mandates.invoke(user_id)
        if not mandates:
            return f"## 🏦 **Mandate History**\n\n❌ No mandates found for client ID: {user_id}"
        
        # Get last 5 records
        recent_mandates = mandates[-5:] if len(mandates) > 5 else mandates
        
        # Calculate summary stats
        total_mandates = len(mandates)
        active_count = len([m for m in mandates if m.get('debicheck_mandate_state') == 'Active'])
        authenticated_count = len([m for m in mandates if m.get('authenticated')])
        
        history = f"""## 🏦 **Mandate History**

### 📊 **Summary**
| Metric | Value |
|:-------|:------|
| 📈 **Total Mandates** | `{total_mandates}` |
| 🟢 **Active** | `{active_count}` |
| ✅ **Authenticated** | `{authenticated_count}` |

### 📋 **Recent Mandates** (Last 5)
| Date | State | Amount | Auth | Service |
|:-----|:------|:-------|:-----|:--------|"""
        
        for mandate in recent_mandates:
            state = mandate.get('debicheck_mandate_state', 'Unknown')
            amount = f"R {float(mandate.get('collection_amount', 0)):,.2f}"
            date = mandate.get('create_ts', 'N/A')[:10] if mandate.get('create_ts') else 'N/A'
            auth = '✅' if mandate.get('authenticated') else '❌'
            service = mandate.get('service', 'N/A')
            
            # Add status emoji
            if state == 'Active':
                state_display = f"🟢 {state}"
            elif state == 'Created':
                state_display = f"🟡 {state}"
            elif state == 'Cancelled':
                state_display = f"🔴 {state}"
            else:
                state_display = f"ℹ️ {state}"
            
            history += f"\n| `{date}` | {state_display} | `{amount}` | {auth} | `{service}` |"
        
        return history
        
    except Exception as e:
        return f"## 🏦 **Mandate History**\n\n❌ Error loading mandates: {str(e)}"


def start_new_conversation() -> str:
    """Start new conversation with fresh thread_id."""
    new_thread_id = str(uuid.uuid4())
    console_capture.write(f"\n🆕 NEW CONVERSATION - Thread: {new_thread_id}\n")
    console_capture.write("🧹 History cleared\n")
    return new_thread_id


def clear_console() -> str:
    """Clear console output."""
    console_capture.clear()
    console_capture.write(f"🧹 Console cleared at {datetime.now().strftime('%H:%M:%S')}\n")
    return console_capture.get_output()


def create_voice_chat_test_app():
    """Create voice chat test app with real-time console updates."""
    
    interface = VoiceTestInterface()
    
    # Clean CSS for better UI
    app_css = """
    .client-info {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        height: 600px;
        overflow-y: auto;
    }
    
    .mandate-info {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 16px;
        height: 600px;
        overflow-y: auto;
    }
    
    .conversation-display {
        background: #0f172a;
        color: #f1f5f9;
        font-family: 'Courier New', monospace;
        border-radius: 8px;
        border: 1px solid #334155;
        height: 300px;
    }
    
    .conversation-display textarea {
        background: #0f172a !important;
        color: #f1f5f9 !important;
        font-family: 'Courier New', monospace !important;
        border: none !important;
        font-size: 13px !important;
    }
    
    .console-display {
        background: #1e293b;
        color: #e2e8f0;
        font-family: 'Courier New', monospace;
        border-radius: 8px;
        border: 1px solid #334155;
        height: 400px;
    }
    
    .console-display textarea {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        font-family: 'Courier New', monospace !important;
        border: none !important;
        font-size: 11px !important;
    }
    
    .status-row {
        background: #f1f5f9;
        border-radius: 6px;
        padding: 8px;
        margin: 4px 0;
    }
    """
    
    with gr.Blocks(title="🎙️ Voice Chat Test", theme=gr.themes.Default(), css=app_css) as app:
        
        gr.Markdown("# 🎙️ **Voice Chat Test Console**")
        gr.Markdown("### Real-time voice testing with client verification and workflow debugging")
        
        # State variables
        current_user_id = gr.State("")
        current_thread_id = gr.State(str(uuid.uuid4()))
        
        # First row: Controls + Voice Interface
        with gr.Row(elem_classes=["status-row"]):
            # Client ID input
            user_id_input = gr.Textbox(
                label="🆔 Client ID",
                placeholder="Enter client ID",
                value="28173",
                scale=2
            )
            
            # Buttons
            with gr.Column(scale=1):
                load_btn = gr.Button("🔄 Load Client", variant="primary")
                new_conv_btn = gr.Button("🆕 New Conversation")
            
            # Status
            status_display = gr.Textbox(
                label="📊 Status",
                value="Ready to load client",
                interactive=False,
                scale=3
            )
            
            # Voice Interface
            audio = WebRTC(
                label="🎤 Voice Communication",
                mode="send-receive",
                modality="audio",
                button_labels={"start": "🎙️ Start Talking", "stop": "⏹️ Stop"},
                track_constraints={
                    "echoCancellation": {"exact": True},
                    "noiseSuppression": {"ideal": True},
                    "autoGainControl": {"exact": True},
                    "sampleRate": {"ideal": 16000},
                    "channelCount": {"exact": 1},
                },
                scale=2
            )
        
        # Second row: 3 Columns - Client Info (1) + Mandates (1) + Console (2)
        with gr.Row():
            # Column 1: Client Information for Verification
            with gr.Column(scale=1):
                client_info_display = gr.Markdown(
                    value="## 👤 **Client Information**\n\nEnter a client ID to load information for verification.",
                    elem_classes=["client-info"],
                    container=True
                )
            
            # Column 2: Mandate History
            with gr.Column(scale=1):
                mandate_history_display = gr.Markdown(
                    value="## 🏦 **Mandate History**\n\nNo mandate history available - load a client first.",
                    elem_classes=["mandate-info"],
                    container=True
                )
            
            # Column 3: Console Area (Double width for more space)
            with gr.Column(scale=2):
                # Conversation Console (Live conversation display)
                gr.Markdown("### 💬 **Live Conversation**")
                
                conversation_output = gr.Textbox(
                    label="Conversation Log",
                    value="💬 Conversation will appear here during voice chat...\n",
                    lines=15,
                    interactive=False,
                    max_lines=20,
                    elem_classes=["conversation-display"],
                    autoscroll=True,
                    show_copy_button=True
                )
                
                # Debug Console (In Accordion)
                with gr.Accordion("🔧 Debug Console", open=False):
                    console_output = gr.Textbox(
                        label="Debug Logs",
                        value="🖥️ Debug console - Load a client to start...\n",
                        lines=20,
                        interactive=False,
                        max_lines=25,
                        elem_classes=["console-display"],
                        autoscroll=True,
                        show_copy_button=True
                    )
                    
                    # Console controls
                    with gr.Row():
                        clear_btn = gr.Button("🧹 Clear Debug", size="sm", variant="secondary")
                        refresh_btn = gr.Button("🔄 Refresh", size="sm", variant="secondary")
                        clear_conv_btn = gr.Button("💬 Clear Conversation", size="sm", variant="secondary")
        
        # Real-time console update functions
        def update_console_realtime():
            """Update debug console output in real-time."""
            return console_capture.get_output()
        
        def update_conversation_realtime():
            """Update conversation console output in real-time."""
            return console_capture.get_conversation_output()
        
        # Set up real-time console updates (reduced frequency to prevent conflicts)
        console_timer = gr.Timer(2.0)  # Update every 2 seconds to reduce conflicts
        console_timer.tick(
            fn=update_console_realtime,
            outputs=[console_output]
        )
        
        # Set up real-time conversation updates (reduced frequency)
        conversation_timer = gr.Timer(1.5)  # Update every 1.5 seconds to reduce conflicts
        conversation_timer.tick(
            fn=update_conversation_realtime,
            outputs=[conversation_output]
        )
        
        # Event handlers
        def load_client_data(user_id, thread_id):
            console_capture.write(f"\n{'='*60}\n")
            console_capture.write(f"📋 LOADING CLIENT DATA\n")
            console_capture.write(f"🆔 Client ID: {user_id}\n")
            console_capture.write(f"🧵 Thread: {thread_id}\n")
            console_capture.write(f"{'='*60}\n")
            
            client_info = get_client_info_display(user_id)
            mandate_history = get_mandate_history(user_id)
            status = interface.update_client_data(user_id)
            
            console_capture.write(f"✅ Client data loaded successfully\n")
            console_capture.write(f"🔍 Ready for voice verification\n\n")
            
            return client_info, mandate_history, status, user_id
        
        load_btn.click(
            fn=load_client_data,
            inputs=[user_id_input, current_thread_id],
            outputs=[client_info_display, mandate_history_display, status_display, current_user_id]
        )
        
        user_id_input.submit(
            fn=load_client_data,
            inputs=[user_id_input, current_thread_id],
            outputs=[client_info_display, mandate_history_display, status_display, current_user_id]
        )
        
        # New conversation
        def start_new_conversation_handler():
            new_thread_id = start_new_conversation()
            return new_thread_id
        
        new_conv_btn.click(
            fn=start_new_conversation_handler,
            outputs=[current_thread_id]
        )
        
        # Audio streaming with conversation logging
        audio.stream(
            ReplyOnPause(
                interface.process_audio_input,
                input_sample_rate=16000,
                algo_options=AlgoOptions(
                    audio_chunk_duration=0.6,
                    started_talking_threshold=0.4,
                    speech_threshold=0.3,
                ),
                model_options=SileroVadOptions(
                    threshold=0.7,
                    min_speech_duration_ms=200,
                    min_silence_duration_ms=1000,
                    window_size_samples=512,
                    speech_pad_ms=50,
                    max_speech_duration_s=10.0,
                ),
            ),
            inputs=[audio, gr.State([]), current_thread_id],
            outputs=[audio],
        )
        
        # Console controls
        def clear_console_handler():
            result = clear_console()
            return result
        
        def clear_conversation_handler():
            console_capture.conversation_output = ""
            console_capture.write("💬 Conversation log cleared\n")
            return "💬 Conversation cleared...\n"
        
        clear_btn.click(
            fn=clear_console_handler,
            outputs=[console_output]
        )
        
        clear_conv_btn.click(
            fn=clear_conversation_handler,
            outputs=[conversation_output]
        )
        
        refresh_btn.click(
            fn=lambda: console_capture.get_output(),
            outputs=[console_output]
        )
        
        # Initialize app
        def initialize():
            console_capture.write(f"🚀 Voice Chat Test Console Initialized\n")
            console_capture.write(f"🕒 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            console_capture.write(f"{'='*60}\n\n")
            
            initial_thread_id = str(uuid.uuid4())
            client_info = get_client_info_display("28173")
            mandate_history = get_mandate_history("28173")
            status = interface.update_client_data("28173")
            return client_info, mandate_history, status, "28173", initial_thread_id
        
        app.load(
            fn=initialize,
            outputs=[client_info_display, mandate_history_display, status_display, current_user_id, current_thread_id]
        )
    
    return app


# Usage
if __name__ == "__main__":
    demo = create_voice_chat_test_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )