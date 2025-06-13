# src/FrontEnd/voice_chat_test_app.py
"""
Voice Chat Test App with FastRTC Integration and Live Call Step Updates
3 Columns: Client Info | Mandate History | Console Logs with Real-time Updates
UPDATED VERSION: Added tool message display and text input for manual messaging
"""

import gradio as gr
import sys,os
import logging
import io
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Generator, Callable, Union, Set

sys.path.append("../..")

from fastrtc import WebRTC, ReplyOnPause, AlgoOptions, SileroVadOptions, AdditionalOutputs
from src.Agents.call_center_agent.data.client_data_fetcher import get_client_data, format_currency, get_safe_value, clear_cache
from src.VoiceHandler import VoiceInteractionHandler
from src.Agents.graph_call_center_agent import create_call_center_agent
from langchain_ollama import ChatOllama
from app_config import CONFIG
from fastrtc import Stream, get_cloudflare_turn_credentials_async, get_cloudflare_turn_credentials
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())
TOKEN = os.environ.get("HF_TOKEN")

async def get_credentials():
    return await get_cloudflare_turn_credentials_async(hf_token=TOKEN)

class ConsoleCapture:
    """Captures console output and conversation logs separately with deduplication and tool message support."""
    
    def __init__(self):
        self.output = ""
        self.conversation_output = ""
        self.buffer = io.StringIO()
        self.last_messages = set()
        self.max_tracked = 100
    
    def write(self, text):
        self.buffer.write(text)
        self.output += text
        return len(text)
    
    def log_conversation(self, role: str, content: str):
        """Log conversation messages with timestamp and deduplication."""
        msg_hash = hash(f"{role}:{content}")
        
        if msg_hash in self.last_messages:
            return
        
        self.last_messages.add(msg_hash)
        if len(self.last_messages) > self.max_tracked:
            self.last_messages = set(list(self.last_messages)[self.max_tracked//2:])
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        role_icons = {
            "user": "👤", "human": "👤",
            "assistant": "🤖", "ai": "🤖",
            "tool": "🔧", "tool_call": "🔧", "tool_response": "📋",
            "system": "📍", "error": "❌"
        }
        
        icon = role_icons.get(role.lower(), role.upper()[:1])
        
        # Special formatting for tool messages
        if role.lower() in ["tool", "tool_call", "tool_response"]:
            # Format tool messages with better readability
            if role.lower() == "tool_call":
                conv_msg = f"[{timestamp}] 🔧 TOOL CALL: {content}\n"
            elif role.lower() == "tool_response":
                conv_msg = f"[{timestamp}] 📋 TOOL RESPONSE: {content}\n"
            else:
                conv_msg = f"[{timestamp}] 🔧 TOOL: {content}\n"
        else:
            conv_msg = f"[{timestamp}] {icon} {role.upper()}: {content}\n"
        
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
    
    def clear_conversation_only(self):
        """Clear only conversation output, keep debug logs."""
        self.conversation_output = ""
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
    """Voice test interface with FastRTC integration and live call step tracking."""
    
    def __init__(self):
        self.voice_handler = None
        self.current_client_data = None
        self.current_user_id = ""
        self.current_thread_id = ""
        self.workflow = None
        self.last_known_step = "Not started"
    
    def get_current_call_step(self) -> str:
        """Get current call step from workflow state with better error handling."""
        try:
            if not self.voice_handler:
                return "No voice handler"
            
            # Try to get workflow from voice handler
            workflow = getattr(self.voice_handler, 'workflow', None)
            if not workflow:
                # Try to get from cached workflow
                workflow = self.workflow
            
            if not workflow:
                return "No workflow"
            
            # Try to get state from workflow
            if hasattr(workflow, 'get_state') and self.current_thread_id:
                try:
                    # Get state with thread_id
                    config = {"configurable": {"thread_id": self.current_thread_id}}
                    state = workflow.get_state(config)
                    
                    if state and hasattr(state, 'values') and 'current_step' in state.values:
                        step = state.values['current_step']
                        formatted_step = step.replace('_', ' ').title() if step else "Unknown"
                        if formatted_step != self.last_known_step:
                            self.last_known_step = formatted_step
                            console_capture.write(f"📍 Call step changed: {formatted_step}\n")
                        return formatted_step
                except Exception as e:
                    console_capture.write(f"⚠️ Error getting workflow state: {e}\n")
            
            return self.last_known_step
            
        except Exception as e:
            console_capture.write(f"⚠️ Error in get_current_call_step: {e}\n")
            return "Step error"
    
    def update_client_data(self, user_id: str, preserve_workflow: bool = False) -> str:
        """Update client data and setup voice handler with option to preserve workflow."""
        try:
            if not user_id or not user_id.strip():
                return "❌ No client ID provided"
            
            console_capture.write(f"\n🔄 {'REFRESHING' if preserve_workflow else 'UPDATING'} CLIENT DATA: {user_id}\n")
            
            # Clear cache for fresh data
            clear_cache(user_id)
            
            # Load client data using fast concurrent fetcher
            client_data = get_client_data(user_id)
            if not client_data:
                return f"❌ No data found for client ID: {user_id}"
            
            self.current_client_data = client_data
            self.current_user_id = user_id
            
            if not preserve_workflow:
                # Full setup for new client
                self.last_known_step = "Introduction"
                
                # Create workflow factory
                def workflow_factory(client_data_param):
                    workflow = self.create_voice_workflow(client_data_param)
                    self.workflow = workflow  # Cache workflow reference
                    return workflow
                
                # Create voice handler
                self.voice_handler = VoiceInteractionHandler(CONFIG, workflow_factory)
                
                # Clear existing handlers and add our handler
                self.voice_handler.message_streamer.handlers.clear()
                self.voice_handler.add_message_handler(console_capture.log_conversation)
            
            # Update client data in existing or new voice handler
            if self.voice_handler:
                self.voice_handler.update_client_data(user_id, client_data)
            
            client_name = get_safe_value(client_data, 'profile.client_info.client_full_name', 'Unknown')
            action = "refreshed" if preserve_workflow else "ready"
            console_capture.write(f"✅ Voice handler {action} for: {client_name}\n")
            
            return f"✅ {'Refreshed' if preserve_workflow else 'Ready for'} client: {client_name}"
            
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
        """Process audio input with live step tracking."""
        if audio_input is None:
            yield audio_input
            return
            
        try:
            # Update thread_id for state tracking
            self.current_thread_id = thread_id
            
            console_capture.write(f"\n{'='*60}\n")
            console_capture.write(f"🎤 PROCESSING AUDIO INPUT\n")
            console_capture.write(f"🧵 Thread ID: {thread_id}\n")
            console_capture.write(f"🕒 Time: {datetime.now().strftime('%H:%M:%S')}\n")
            console_capture.write(f"📍 Current Step: {self.get_current_call_step()}\n")
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
    
    def process_text_input(self, text_input: str, thread_id: str) -> Generator:
        """Process text input through the voice handler with audio streaming."""
        try:
            if not text_input.strip():
                yield "", "📝 Text input is empty", None
                return
            
            # Update thread_id for state tracking
            self.current_thread_id = thread_id
            
            console_capture.write(f"\n{'='*60}\n")
            console_capture.write(f"📝 PROCESSING TEXT INPUT\n")
            console_capture.write(f"🧵 Thread ID: {thread_id}\n")
            console_capture.write(f"🕒 Time: {datetime.now().strftime('%H:%M:%S')}\n")
            console_capture.write(f"📍 Current Step: {self.get_current_call_step()}\n")
            console_capture.write(f"💬 Message: {text_input}\n")
            console_capture.write(f"{'='*60}\n")
            
            if self.voice_handler:
                # Process the message through the workflow using the voice handler's method
                try:
                    # Use the voice handler's process_text_input method which handles TTS
                    for text_result in self.voice_handler.process_text_input(
                        text_input, None, [], thread_id
                    ):
                        if text_result is None:
                            # Still processing
                            yield "", "🔄 Processing...", None
                        else:
                            audio_chunk, chatbot, _ = text_result
                            if audio_chunk is not None:
                                # Audio chunk received, stream it
                                yield "", "🔊 Playing response...", audio_chunk
                            else:
                                # Final result
                                yield "", "✅ Message processed successfully", None
                
                except Exception as e:
                    console_capture.write(f"❌ Text processing error: {e}\n")
                    yield "", f"❌ Error: {str(e)}", None
            else:
                console_capture.write("❌ Voice handler not available\n")
                yield "", "❌ Voice handler not available", None
                
        except Exception as e:
            console_capture.write(f"❌ Text processing error: {e}\n")
            yield "", f"❌ Error: {str(e)}", None


def safe_format_currency(value) -> str:
    """Safely format currency values."""
    try:
        if value is None or value == '':
            return "R 0.00"
        num_value = float(str(value).replace(',', '').replace('R', '').strip())
        return f"R {num_value:.2f}"
    except (ValueError, TypeError, AttributeError):
        return "R 0.00"


def get_client_info_display(user_id: str) -> str:
    """Get formatted client info for verification using cached data fetcher."""
    try:
        if not user_id or not user_id.strip():
            return "## 👤 **Client Information**\n\nEnter a client ID to load information for verification."
        
        client_data = get_client_data(user_id)
        if not client_data:
            return f"## ❌ **Client Not Found**\n\nNo data available for client ID: {user_id}"
        
        # Extract key info
        client_name = get_safe_value(client_data, 'profile.client_info.client_full_name', 'Unknown')
        title = get_safe_value(client_data, 'profile.client_info.title', 'Mr/Ms')
        id_number = get_safe_value(client_data, 'profile.client_info.id_number', 'N/A')
        email = get_safe_value(client_data, 'profile.client_info.email_address', 'N/A')
        username = get_safe_value(client_data, 'profile.user_name', 'N/A')
        
        # Calculate outstanding amount
        account_aging = client_data.get("account_aging", {})
        try:
            total = float(str(account_aging.get("xbalance", 0)).replace(',', '').replace('R', '').strip() or 0)
            current = float(str(account_aging.get("x0", 0)).replace(',', '').replace('R', '').strip() or 0)
            outstanding = max(total - current, 0.0)
            outstanding_str = safe_format_currency(outstanding)
        except:
            outstanding_str = "R 0.00"
        
        # Vehicle info
        vehicles = get_safe_value(client_data, 'profile.vehicles', [])
        vehicle_info = "🚫 No vehicles"
        if vehicles and len(vehicles) > 0:
            vehicle = vehicles[0]
            reg = vehicle.get('registration', 'N/A')
            make = vehicle.get('make', 'N/A')
            model = vehicle.get('model', 'N/A')
            color = vehicle.get('color', 'N/A')
            vehicle_info = f"🚗 {reg} - {make} {model} ({color})"
        
        # Format aging amounts
        amounts = {
            'total_balance': safe_format_currency(account_aging.get('xbalance', 0)),
            'current': safe_format_currency(account_aging.get('x0', 0)),
            'x30': safe_format_currency(account_aging.get('x30', 0)),
            'x60': safe_format_currency(account_aging.get('x60', 0)),
            'x90': safe_format_currency(account_aging.get('x90', 0)),
            'x120': safe_format_currency(account_aging.get('x120', 0))
        }
        
        return f"""## 👤 **{title} {client_name}**

### 🔍 **Verification Details**
| Field | Value |
|:------|:------|
| 🆔 **User ID** | `{user_id}` |
| 📄 **ID Number** | `{id_number}` |
| 📧 **Email** | `{email}` |
| 👤 **Username** | `{username}` |
| 💰 **Outstanding** | **`{outstanding_str}`** |
| 📊 **Total Balance** | `{amounts['total_balance']}` |

### 🚗 **Vehicle Information**
{vehicle_info}

### 📅 **Account Aging Breakdown**
| Period | Amount |
|:-------|:-------|
| 🟢 **Current** | `{amounts['current']}` |
| 🟡 **30 Days** | `{amounts['x30']}` |
| 🟠 **60 Days** | `{amounts['x60']}` |
| 🔴 **90 Days** | `{amounts['x90']}` |
| ⚠️ **120+ Days** | `{amounts['x120']}` |

---
*Use this information to verify client identity during the call*
"""
    
    except Exception as e:
        return f"## ❌ **Error Loading Client Data**\n\nError: {str(e)}"


def get_mandate_history(user_id: str) -> str:
    """Get formatted payment/banking summary from cached client data."""
    try:
        if not user_id or not user_id.strip():
            return "## 🏦 **Payment Summary**\n\nNo payment data available - load a client first."
        
        client_data = get_client_data(user_id)
        if not client_data:
            return f"## 🏦 **Payment Summary**\n\n❌ No client data found for ID: {user_id}"
        
        banking_details = client_data.get('banking_details', {})
        payment_history = client_data.get('payment_history', [])
        
        summary_content = """## 🏦 **Payment & Banking Summary**

### 📊 **Banking Information**
| Field | Value |
|:------|:------|"""
        
        if banking_details:
            bank_name = banking_details.get('bank_name', 'N/A')
            account_type = banking_details.get('bank_account_type', 'N/A')
            payment_method = banking_details.get('payment_method', 'N/A')
            debit_date = banking_details.get('debit_date', 'N/A')
            allow_debit = '✅ Yes' if banking_details.get('allow_debit') else '❌ No'
            
            summary_content += f"""
| 🏦 **Bank** | `{bank_name}` |
| 💳 **Account Type** | `{account_type}` |
| 💰 **Payment Method** | `{payment_method}` |
| 📅 **Debit Date** | `{debit_date}` |
| ✅ **Allow Debit** | {allow_debit} |"""
        else:
            summary_content += "\n| ❌ **No banking details** | Available |"
        
        # Payment history summary
        if payment_history:
            total_arrangements = len(payment_history)
            recent_arrangements = payment_history[-3:] if len(payment_history) > 3 else payment_history
            
            summary_content += f"""

### 📋 **Recent Payment Arrangements** (Last 3 of {total_arrangements})
| Date | Amount | Status | Type |
|:-----|:-------|:-------|:-----|"""
            
            for arrangement in recent_arrangements:
                date = arrangement.get('create_ts', 'N/A')[:10] if arrangement.get('create_ts') else 'N/A'
                amount = safe_format_currency(arrangement.get('amount', 0))
                state = arrangement.get('arrangement_state', 'Unknown')
                pay_type = arrangement.get('arrangement_pay_type', 'N/A')
                
                # Status emojis
                status_emojis = {
                    'Successful': '🟢', 'Failed': '🔴', 'Pending': '🟡'
                }
                state_display = f"{status_emojis.get(state, 'ℹ️')} {state}"
                
                summary_content += f"\n| `{date}` | `{amount}` | {state_display} | `{pay_type}` |"
        else:
            summary_content += "\n\n### 📋 **Payment Arrangements**\n❌ No payment arrangement history available"
        
        return summary_content
        
    except Exception as e:
        return f"## 🏦 **Payment Summary**\n\n❌ Error loading data: {str(e)}"


def start_new_conversation() -> Tuple[str, str]:
    """Start new conversation with fresh thread_id and clear conversation."""
    # Generate truly unique thread ID with timestamp and random component
    timestamp = str(int(datetime.now().timestamp() * 1000))  # Millisecond timestamp
    random_part = str(uuid.uuid4())[:8]
    new_thread_id = f"thread_{timestamp}_{random_part}"
    
    console_capture.write(f"\n🆕 NEW CONVERSATION - Thread: {new_thread_id}\n")
    console_capture.write("🧹 Conversation history cleared\n")
    
    # Clear conversation logs but keep debug logs
    console_capture.clear_conversation_only()
    
    return new_thread_id, "💬 New conversation started...\n"


def clear_console() -> str:
    """Clear console output."""
    console_capture.clear()
    console_capture.write(f"🧹 Console cleared at {datetime.now().strftime('%H:%M:%S')}\n")
    return console_capture.get_output()


def create_voice_chat_test_app():
    """Create voice chat test app with live call step updates and text input."""
    
    interface = VoiceTestInterface()
    user_id = "1489698"
    # Clean CSS for better UI
    app_css = """
    .client-info {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
    }
    
    .mandate-info {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 16px;
    }
    
    .conversation-display textarea {
        background: #0f172a !important;
        color: #f1f5f9 !important;
        font-family: 'Courier New', monospace !important;
        border: none !important;
        font-size: 13px !important;
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
    
    .text-input-section {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 12px;
        margin-top: 8px;
    }
    """
    
    with gr.Blocks(title="🎙️ Call Center AI Agent", theme=gr.themes.Default(), css=app_css) as app:
        
        gr.Markdown("# 🎙️ **Debtor Call Center AI Agent**")
        gr.Markdown("### Real-time voice call testing with live call step tracking and tool message display")
        
        # State variables - Generate unique thread_id for each session
        session_thread_id = str(uuid.uuid4())  # Unique per page load/visit
        current_user_id = gr.State("")
        current_thread_id = gr.State(session_thread_id)
        
        # Enhanced audio constraints
        enhanced_constraints = {
            "echoCancellation": {"exact": True},
            "noiseSuppression": {"exact": True},
            "autoGainControl": {"exact": True},
            "sampleRate": {"ideal": 16000},
            "channelCount": {"exact": 1},
            
            # Advanced noise suppression
            "googNoiseSuppression": {"exact": True},
            "googNoiseSuppression2": {"exact": True},
            "googEchoCancellation": {"exact": True},
            "googEchoCancellation2": {"exact": True},
            "googAutoGainControl": {"exact": True},
            "googHighpassFilter": {"exact": True},
            "googTypingNoiseDetection": {"exact": True}
        }
        
        # First row: Controls Layout
        with gr.Row(elem_classes=["status-row"]):
            # Voice Interface
            with gr.Column(scale=1):
                audio = WebRTC(
                    label="🎤 Voice Communication",
                    mode="send-receive",
                    modality="audio",
                    button_labels={"start": "🎙️ Start", "stop": "⏹️ Stop"},
                    track_constraints=enhanced_constraints,
                    rtc_configuration=get_credentials,
                    server_rtc_configuration=get_cloudflare_turn_credentials(ttl=360_000),
                    min_width=80,
                )

            
            # Middle: Status Information
            with gr.Column(scale=1):
                with gr.Row():
                    script_type_display = gr.Textbox(
                        label="📋 Script Type",
                        value="Not determined",
                        interactive=False,
                        scale=1
                    )
                    call_step_display = gr.Textbox(
                        label="📍 Call Step",
                        value="Not started",
                        interactive=False,
                        scale=1
                    )
                with gr.Row():
                    new_conv_btn = gr.Button("🆕 New conversation", variant="primary", size="sm")
                    refresh_data_btn = gr.Button("🔄 Refresh Debtor Data", variant="secondary", size="sm")
                    
            
            # Left: Client ID and Buttons
            with gr.Column(scale=1):
                with gr.Row():
                    user_id_input = gr.Textbox(
                        label="🆔 Client ID",
                        placeholder="Enter client ID",
                        value=user_id,
                        scale=3
                    )
                       
                    status_display = gr.Textbox(
                        label="📊 Status",
                        value="Ready to load client",
                        interactive=False,
                        scale=1
                    )
                load_btn = gr.Button("Search debtor data", variant="primary", size="sm")
            
        
        # Second row: 3 Columns
        with gr.Row():
            # Column 3: Console Area
            with gr.Column(scale=2):
                # Live Conversation
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### 💬 **Live Conversation & Tool Messages**")
                    # Console controls
                    clear_btn = gr.Button("🧹 Clear Debug", size="sm", variant="secondary", scale=1)
                    clear_conv_btn = gr.Button("💬 Clear Conversation", size="sm", variant="secondary", scale=1)
                    
                
                conversation_output = gr.Textbox(
                    label="Conversation Log (includes tool calls and responses)",
                    value="💬 Conversation will appear here during voice chat...\n🔧 Tool messages will also be displayed here\n",
                    lines=15,
                    interactive=False,
                    max_lines=15,
                    elem_classes=["conversation-display"],
                    autoscroll=True,
                    show_copy_button=True
                )
                
                # Debug Console
                with gr.Accordion("🔧 Debug Console", open=True):
                    console_output = gr.Textbox(
                        label="Debug Logs",
                        value="🖥️ Debug console - Load a client to start...\n",
                        lines=12,
                        interactive=False,
                        max_lines=12,
                        elem_classes=["console-display"],
                        autoscroll=True,
                        show_copy_button=True
                    )
                
                # Text Input Section
                with gr.Group(elem_classes=["text-input-section"]):
                    gr.Markdown("### 📝 **Manual Text Input**")
                    gr.Markdown("*Send text messages directly to the agent (bypasses voice processing)*")
                    
                    with gr.Row():
                        text_input = gr.Textbox(
                            label="",
                            placeholder="Type your message here and press Enter or click Send...",
                            scale=4,
                            lines=1,
                            max_lines=3
                        )
                        send_btn = gr.Button("📤 Send", variant="primary", size="sm", scale=1)
                    
                    text_status = gr.Textbox(
                        label="Text Status",
                        value="Ready to send text messages",
                        interactive=False,
                        lines=1
                    )
                    
                    # Hidden audio component for text chat TTS output
                    text_chat_audio = gr.Audio(
                        label="", 
                        autoplay=True,
                        visible=False,
                        interactive=False,
                        streaming=True,
                        elem_id="response-audio"
                    )
                    
                    
            # Column 1: Client Information
            with gr.Column(scale=1):
                client_info_display = gr.Markdown(
                    value="## 👤 **Client Information**\n\nEnter a client ID to load information for verification.",
                    elem_classes=["client-info"],
                    container=True
                )
                
            
            # Column 2: Payment Summary
            with gr.Column(scale=1):
                mandate_history_display = gr.Markdown(
                    value="## 🏦 **Payment Summary**\n\nNo payment data available - load a client first.",
                    elem_classes=["mandate-info"],
                    container=True
                )
        
        # Real-time updates with proper error handling
        def update_console_realtime():
            try:
                return console_capture.get_output()
            except:
                return "Console update error"
        
        def update_conversation_realtime():
            try:
                return console_capture.get_conversation_output()
            except:
                return "Conversation update error"
        
        def update_call_step_realtime():
            try:
                return interface.get_current_call_step()
            except:
                return "Step update error"
        
        # Timers for real-time updates
        console_timer = gr.Timer(2.0)
        console_timer.tick(fn=update_console_realtime, outputs=[console_output])
        
        conversation_timer = gr.Timer(1.5)
        conversation_timer.tick(fn=update_conversation_realtime, outputs=[conversation_output])
        
        # Live call step updates
        step_timer = gr.Timer(1.0)
        step_timer.tick(fn=update_call_step_realtime, outputs=[call_step_display])
        
        # Event handlers
        def load_client_data(user_id, thread_id):
            """Load client data and determine script type."""
            console_capture.write(f"\n{'='*60}\n")
            console_capture.write(f"📋 LOADING CLIENT DATA\n")
            console_capture.write(f"🆔 Client ID: {user_id}\n")
            console_capture.write(f"🧵 Thread: {thread_id}\n")
            console_capture.write(f"{'='*60}\n")
            
            # Load client data and determine script type
            client_data = get_client_data(user_id)
            script_display = "Not determined"
            
            if client_data:
                try:
                    from src.Agents.call_center_agent.call_scripts import ScriptManager
                    account_aging = client_data.get("account_aging", {})
                    script_type_enum = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
                    script_type = script_type_enum.value if hasattr(script_type_enum, 'value') else str(script_type_enum)
                    script_display = script_type.replace('_', ' ').title()
                    console_capture.write(f"📋 Script Type: {script_display}\n")
                except Exception as e:
                    console_capture.write(f"⚠️ Could not determine script type: {e}\n")
                    script_display = "Script determination failed"
            
            # Update interface
            interface.current_thread_id = thread_id  # Set thread_id for live tracking
            
            client_info = get_client_info_display(user_id)
            mandate_history = get_mandate_history(user_id)
            status = interface.update_client_data(user_id, preserve_workflow=False)
            
            console_capture.write(f"✅ Client data loaded successfully\n")
            console_capture.write(f"🔍 Ready for voice verification\n\n")
            
            return client_info, mandate_history, status, user_id, script_display, "Introduction"
        
        def refresh_client_data_only(user_id):
            """Refresh client data without affecting workflow or conversation."""
            if not user_id:
                return get_client_info_display(""), get_mandate_history(""), "❌ No client ID"
            
            console_capture.write(f"\n📊 REFRESHING CLIENT DATA (preserving workflow)\n")
            
            client_info = get_client_info_display(user_id)
            mandate_history = get_mandate_history(user_id)
            status = interface.update_client_data(user_id, preserve_workflow=True)
            
            console_capture.write(f"📊 Client data refreshed without disrupting conversation\n\n")
            
            return client_info, mandate_history, status
        
        def send_text_message(text_input_value, thread_id):
            """Send text message through the voice handler with audio streaming."""
            if not text_input_value or not text_input_value.strip():
                yield "", "❌ Please enter a message", None
                return
            
            try:
                for result in interface.process_text_input(text_input_value.strip(), thread_id):
                    cleared_input, status, audio_chunk = result
                    yield cleared_input, status, audio_chunk
            except Exception as e:
                yield text_input_value, f"❌ Error: {str(e)}", None
        
        # Connect events
        load_btn.click(
            fn=load_client_data,
            inputs=[user_id_input, current_thread_id],
            outputs=[client_info_display, mandate_history_display, status_display, 
                    current_user_id, script_type_display, call_step_display]
        )
        
        user_id_input.submit(
            fn=load_client_data,
            inputs=[user_id_input, current_thread_id],
            outputs=[client_info_display, mandate_history_display, status_display, 
                    current_user_id, script_type_display, call_step_display]
        )
        
        # NEW: Refresh Data Button Handler
        refresh_data_btn.click(
            fn=refresh_client_data_only,
            inputs=[current_user_id],
            outputs=[client_info_display, mandate_history_display, status_display]
        )
        
        # Text input handlers - now with streaming audio support
        send_btn.click(
            fn=send_text_message,
            inputs=[text_input, current_thread_id],
            outputs=[text_input, text_status, text_chat_audio]
        )
        
        text_input.submit(
            fn=send_text_message,
            inputs=[text_input, current_thread_id],
            outputs=[text_input, text_status, text_chat_audio]
        )
        
        # New conversation handler - now clears conversation
        new_conv_btn.click(
            fn=start_new_conversation,
            outputs=[current_thread_id, conversation_output]
        )
        
        # Audio streaming
        audio.stream(
            ReplyOnPause(
                interface.process_audio_input,
                input_sample_rate=16000,
                algo_options=AlgoOptions(
                    audio_chunk_duration=0.6,
                    started_talking_threshold=0.3,
                    speech_threshold=0.2,
                ),
                model_options=SileroVadOptions(
                    threshold=0.6,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=500,
                ),
                can_interrupt=False,
            ),
            inputs=[audio, gr.State([]), current_thread_id],
            outputs=[audio],
        )
        
        # Console controls
        def clear_conversation_handler():
            console_capture.conversation_output = ""
            console_capture.write("💬 Conversation log cleared\n")
            return "💬 Conversation cleared...\n"
        
        clear_btn.click(fn=clear_console, outputs=[console_output])
        
        clear_conv_btn.click(
            fn=clear_conversation_handler,
            outputs=[conversation_output]
        )
        
        # Initialize app
        def initialize():
            console_capture.write(f"🚀 Voice Chat Test Console Initialized\n")
            console_capture.write(f"🕒 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            console_capture.write(f"⚡ Using fast concurrent data fetcher\n")
            console_capture.write(f"📍 Live call step tracking enabled\n")
            console_capture.write(f"🔄 Refresh data button added (preserves workflow)\n")
            console_capture.write(f"🔧 Tool message display enabled\n")
            console_capture.write(f"📝 Text input feature added with audio streaming\n")
            console_capture.write(f"🧵 Session Thread ID: {session_thread_id}\n")
            console_capture.write(f"{'='*60}\n\n")
            
            # Use the session thread ID that was created at page load
            interface.current_thread_id = session_thread_id
            
            # Load initial client data and determine script type
            client_data = get_client_data(user_id)
            script_display = "Not determined"
            
            if client_data:
                try:
                    from src.Agents.call_center_agent.call_scripts import ScriptManager
                    account_aging = client_data.get("account_aging", {})
                    script_type_enum = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
                    script_type = script_type_enum.value if hasattr(script_type_enum, 'value') else str(script_type_enum)
                    script_display = script_type.replace('_', ' ').title()
                except Exception as e:
                    console_capture.write(f"⚠️ Initial script type determination failed: {e}\n")
                    script_display = "Script determination failed"
            
            client_info = get_client_info_display(user_id)
            mandate_history = get_mandate_history(user_id)
            status = interface.update_client_data(user_id)
            
            return (client_info, mandate_history, status, user_id, 
                   session_thread_id, script_display, "Introduction", "Ready to send text messages")
        
        app.load(
            fn=initialize,
            outputs=[client_info_display, mandate_history_display, status_display, 
                    current_user_id, current_thread_id, script_type_display, call_step_display, text_status]
        )
    
    return app


# Usage
if __name__ == "__main__":
    demo = create_voice_chat_test_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )