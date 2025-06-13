"""
Voice Chat Test App with LiveKit Integration (Self-Hosted)
Maintains existing UI design and functionality while using LiveKit backend
No API keys required - connects to self-hosted LiveKit server
"""

import gradio as gr
import sys
import logging
import io
import uuid
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Generator, Callable, Union, Set

sys.path.append("../..")

from src.Agents.call_center_agent.data.client_data_fetcher import get_client_data, format_currency, get_safe_value, clear_cache
from src.VoiceHandler import VoiceInteractionHandler  # Now uses LiveKit implementation
from src.Agents.graph_call_center_agent import create_call_center_agent
from langchain_ollama import ChatOllama
from app_config import CONFIG


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
    """Voice test interface with LiveKit integration and live call step tracking."""
    
    def __init__(self):
        self.voice_handler = None
        self.current_client_data = None
        self.current_user_id = ""
        self.current_thread_id = ""
        self.workflow = None
        self.last_known_step = "Not started"
        self.livekit_room = None
        self.is_connected = False
    
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
            if not user_id:
                return "❌ Please enter a valid User ID"
            
            console_capture.write(f"📊 Fetching data for User ID: {user_id}\n")
            
            # Clear cache to ensure fresh data
            clear_cache()
            
            # Fetch client data using concurrent fetcher
            client_data = get_client_data(user_id, use_parallel=True)
            
            if not client_data or "error" in client_data:
                error_msg = client_data.get("error", "Unknown error") if client_data else "No data found"
                console_capture.write(f"❌ Failed to fetch client data: {error_msg}\n")
                return f"❌ Error: {error_msg}"
            
            self.current_client_data = client_data
            self.current_user_id = user_id
            
            # Create new thread ID for this session
            self.current_thread_id = f"thread_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Initialize voice handler if not already done or if preserving workflow
            if not self.voice_handler:
                console_capture.write("🎙️ Initializing LiveKit voice handler...\n")
                config = CONFIG.copy()
                config['configurable'] = {'thread_id': self.current_thread_id}
                config['livekit'] = {
                    'url': 'ws://localhost:7880',  # Self-hosted LiveKit server
                    'use_cloud': False,
                    'room_name': f'voice-chat-{user_id}-{uuid.uuid4().hex[:8]}',
                    'participant_name': 'AI-Assistant'
                }
                
                # Create LLM for workflow
                console_capture.write("🧠 Creating AI model and workflow...\n")
                llm = ChatOllama(model='qwen2.5:7b', temperature=0.5)
                
                def workflow_factory(client_data):
                    return create_call_center_agent(
                        model=llm,
                        client_data=client_data,
                        agent_name="Qwen",
                        config=config,
                        verbose=True
                    )
                
                self.voice_handler = VoiceInteractionHandler(config, workflow_factory)
                
                # Set up message handler
                def message_handler(message: Dict[str, Any]):
                    role = message.get('role', 'system')
                    content = message.get('content', '')
                    console_capture.log_conversation(role, content)
                
                self.voice_handler.add_message_handler(message_handler)
            
            # Update client data and workflow
            if not preserve_workflow or not self.workflow:
                console_capture.write("🔄 Creating workflow for user...\n")
                self.voice_handler.set_client_data(client_data, user_id)
                self.workflow = self.voice_handler.get_current_workflow()
            else:
                console_capture.write("♻️ Preserving existing workflow state\n")
            
            console_capture.write(f"✅ Client data loaded successfully for {user_id}\n")
            console_capture.write(f"🧵 Thread ID: {self.current_thread_id}\n")
            
            return "✅ Client data loaded successfully"
            
        except Exception as e:
            console_capture.write(f"❌ Error updating client data: {str(e)}\n")
            return f"❌ Error: {str(e)}"
    
    async def connect_to_livekit(self) -> Tuple[str, str]:
        """Connect to LiveKit server and get connection info."""
        try:
            if not self.voice_handler:
                return "", "❌ No voice handler initialized"
            
            # Connect to LiveKit room
            room_name = await self.voice_handler.connect_to_room()
            self.livekit_room = room_name
            self.is_connected = True
            
            # Generate connection info for client
            connection_info = {
                "url": self.voice_handler.config['livekit']['url'],
                "room": room_name,
                "token": "client-token-placeholder"  # Client will generate proper token
            }
            
            console_capture.write(f"✅ Connected to LiveKit room: {room_name}\n")
            return json.dumps(connection_info), "✅ Connected to LiveKit"
            
        except Exception as e:
            console_capture.write(f"❌ LiveKit connection error: {e}\n")
            return "", f"❌ Connection error: {str(e)}"
    
    async def disconnect_from_livekit(self) -> str:
        """Disconnect from LiveKit server."""
        try:
            if self.voice_handler and self.is_connected:
                await self.voice_handler.disconnect()
                self.is_connected = False
                self.livekit_room = None
                console_capture.write("✅ Disconnected from LiveKit\n")
                return "✅ Disconnected"
            return "❓ Not connected"
        except Exception as e:
            return f"❌ Disconnect error: {str(e)}"
    
    def process_text_input(self, text: str, thread_id: str) -> str:
        """Process text input through the workflow."""
        if not self.voice_handler or not self.workflow:
            console_capture.write("❌ No workflow available. Please select a client first.\n")
            return "No workflow available"
        
        try:
            console_capture.log_conversation("user", text)
            
            # Process through workflow
            result = self.voice_handler.process_message(text, self.workflow)
            
            # Log all message types
            if "messages" in result:
                for msg in result["messages"]:
                    if hasattr(msg, 'content') and hasattr(msg, '__class__'):
                        role = "assistant" if "AI" in msg.__class__.__name__ else "tool"
                        console_capture.log_conversation(role, msg.content)
            
            # Log tool calls
            if "tool_calls" in result:
                for tool_call in result["tool_calls"]:
                    console_capture.log_conversation("tool_call", 
                        f"{tool_call.get('name', 'unknown')}: {tool_call.get('args', {})}")
            
            # Log tool responses
            if "tool_responses" in result:
                for tool_resp in result["tool_responses"]:
                    console_capture.log_conversation("tool_response", 
                        f"{tool_resp.get('name', 'unknown')}: {tool_resp.get('output', 'No output')}")
            
            return result.get("response", "No response generated")
            
        except Exception as e:
            console_capture.write(f"❌ Error processing text: {str(e)}\n")
            return f"Error: {str(e)}"
    
    def get_client_info_display(self) -> str:
        """Format client information for display."""
        if not self.current_client_data:
            return "No client data loaded"
        
        # Format client info (same as original)
        info = []
        data = self.current_client_data
        
        # Client details section
        info.append("### 👤 CLIENT INFORMATION")
        info.append(f"**Name:** {get_safe_value(data, 'client_info.name', 'N/A')}")
        info.append(f"**ID:** {get_safe_value(data, 'client_info.id_number', 'N/A')}")
        info.append(f"**Phone:** {get_safe_value(data, 'client_info.mobile_phone', 'N/A')}")
        info.append(f"**Email:** {get_safe_value(data, 'client_info.email', 'N/A')}")
        info.append(f"**Location:** {get_safe_value(data, 'client_info.physical_city', 'N/A')}")
        info.append("")
        
        # Account status section
        info.append("### 💳 ACCOUNT STATUS")
        info.append(f"**Account #:** {get_safe_value(data, 'account_details.account_number', 'N/A')}")
        info.append(f"**Status:** {get_safe_value(data, 'account_details.account_status', 'N/A')}")
        info.append(f"**Outstanding:** {format_currency(get_safe_value(data, 'payment_info.total_outstanding', 0))}")
        info.append(f"**Overdue:** {format_currency(get_safe_value(data, 'payment_info.overdue_amount', 0))}")
        info.append(f"**Days Overdue:** {get_safe_value(data, 'payment_info.days_overdue', 0)}")
        info.append("")
        
        # Subscription section
        info.append("### 📋 SUBSCRIPTION")
        info.append(f"**Amount:** {format_currency(get_safe_value(data, 'subscription_info.subscription_amount', 0))}")
        info.append(f"**Frequency:** {get_safe_value(data, 'subscription_info.payment_frequency', 'N/A')}")
        info.append(f"**Due Date:** {get_safe_value(data, 'subscription_info.debit_date', 'N/A')}")
        info.append("")
        
        # Vehicle section
        info.append("### 🚗 VEHICLE INFO")
        vehicles = get_safe_value(data, 'vehicles', [])
        if vehicles:
            for vehicle in vehicles[:3]:  # Show first 3 vehicles
                info.append(f"• {vehicle.get('registration', 'N/A')} - {vehicle.get('make', '')} {vehicle.get('model', '')}")
        else:
            info.append("No vehicles found")
        
        return "\n".join(info)
    
    def get_mandate_history_display(self) -> str:
        """Format mandate history for display."""
        if not self.current_client_data:
            return "No mandate history available"
        
        # Format mandate history (same as original)
        info = []
        info.append("### 📑 PAYMENT MANDATE HISTORY")
        info.append("")
        
        # Active mandates
        active_mandates = get_safe_value(self.current_client_data, 'bank_details.active_mandates', [])
        if active_mandates:
            info.append("**✅ ACTIVE MANDATES:**")
            for mandate in active_mandates[:3]:
                info.append(f"• **{mandate.get('type', 'N/A')}** - {mandate.get('bank', 'N/A')} ({mandate.get('status', 'N/A')})")
                info.append(f"  Ref: {mandate.get('reference', 'N/A')}")
                info.append(f"  Created: {mandate.get('created_date', 'N/A')}")
                info.append("")
        else:
            info.append("No active mandates")
            info.append("")
        
        # Recent payment history
        payments = get_safe_value(self.current_client_data, 'payment_history.recent_payments', [])
        if payments:
            info.append("**💰 RECENT PAYMENTS:**")
            for payment in payments[:5]:
                status_emoji = "✅" if payment.get('status') == 'Success' else "❌"
                info.append(f"{status_emoji} {payment.get('date', 'N/A')} - {format_currency(payment.get('amount', 0))}")
                info.append(f"   {payment.get('method', 'N/A')} - {payment.get('reference', 'N/A')}")
                info.append("")
        else:
            info.append("No recent payment history")
            info.append("")
        
        # Failed mandates
        failed_mandates = get_safe_value(self.current_client_data, 'bank_details.failed_mandates', [])
        if failed_mandates:
            info.append("**❌ FAILED MANDATES:**")
            for mandate in failed_mandates[:3]:
                info.append(f"• {mandate.get('type', 'N/A')} - {mandate.get('bank', 'N/A')}")
                info.append(f"  Reason: {mandate.get('failure_reason', 'N/A')}")
                info.append(f"  Date: {mandate.get('failed_date', 'N/A')}")
                info.append("")
        
        # Account notes
        notes = get_safe_value(self.current_client_data, 'account_notes.recent_notes', [])
        if notes:
            info.append("**📝 RECENT NOTES:**")
            for note in notes[:3]:
                info.append(f"• {note.get('date', 'N/A')} - {note.get('note', 'N/A')}")
                info.append(f"  By: {note.get('created_by', 'N/A')}")
                info.append("")
        
        return "\n".join(info)


def create_gradio_interface():
    """Create the Gradio interface with LiveKit integration."""
    interface = VoiceTestInterface()
    
    with gr.Blocks(title="Voice Chat Test - LiveKit", theme=gr.themes.Soft()) as demo:
        # Header
        gr.Markdown("# 🎙️ Voice Chat Test Application - LiveKit Edition")
        gr.Markdown("AI-powered call center agent with LiveKit voice processing")
        
        # Session info
        session_id = str(uuid.uuid4())
        console_capture.write(f"🆔 Session ID: {session_id}\n")
        
        # Hidden LiveKit connection info
        livekit_connection = gr.State("")
        
        with gr.Row():
            # Left Column - Client Information
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Client Selector")
                
                with gr.Row():
                    user_id_input = gr.Textbox(
                        label="User ID",
                        placeholder="Enter user ID (e.g., 61676)",
                        value="61676"
                    )
                    
                    fetch_btn = gr.Button("🔍 Fetch Data", variant="primary")
                    refresh_btn = gr.Button("🔄", variant="secondary", scale=0.2)
                
                fetch_status = gr.Textbox(label="Status", interactive=False)
                current_thread_id = gr.Textbox(label="Thread ID", interactive=False, visible=False)
                
                # Client info display
                client_info_display = gr.Markdown(
                    value="No client data loaded",
                    elem_id="client_info"
                )
            
            # Middle Column - Mandate History
            with gr.Column(scale=1):
                mandate_history_display = gr.Markdown(
                    value="No mandate history available",
                    elem_id="mandate_history"
                )
            
            # Right Column - Console & Controls
            with gr.Column(scale=1):
                # Call status and step display
                with gr.Row():
                    call_step_display = gr.Textbox(
                        label="📍 Current Call Step",
                        value="Not started",
                        interactive=False,
                        elem_id="call_step"
                    )
                
                # LiveKit Controls
                with gr.Row():
                    connect_btn = gr.Button("🔌 Connect Voice", variant="primary")
                    disconnect_btn = gr.Button("🔌 Disconnect", variant="stop")
                    connection_status = gr.Textbox(label="Connection", value="Disconnected", interactive=False)
                
                # Text input for testing
                with gr.Row():
                    text_input = gr.Textbox(
                        label="📝 Text Input (for testing)",
                        placeholder="Type a message...",
                        lines=2
                    )
                    send_btn = gr.Button("📤 Send", variant="secondary")
                
                # Conversation log with tabs
                with gr.Tabs():
                    with gr.Tab("💬 Conversation"):
                        conversation_output = gr.Textbox(
                            label="Conversation Log",
                            lines=15,
                            max_lines=20,
                            interactive=False,
                            elem_id="conversation_log"
                        )
                        clear_conv_btn = gr.Button("🗑️ Clear Conversation", size="sm")
                    
                    with gr.Tab("🖥️ Console"):
                        console_output = gr.Textbox(
                            label="System Console",
                            value=console_capture.get_output(),
                            lines=15,
                            max_lines=20,
                            interactive=False,
                            elem_id="console"
                        )
                        clear_btn = gr.Button("🗑️ Clear Console", size="sm")
                
                # Log levels
                with gr.Row():
                    log_level = gr.Radio(
                        label="Log Level",
                        choices=["none", "error", "warning", "info", "debug"],
                        value="info",
                        elem_id="log_level"
                    )
        
        # LiveKit WebRTC component (hidden, for client-side connection)
        livekit_webrtc = gr.HTML(
            """
            <div id="livekit-container" style="display:none;">
                <audio id="livekit-audio" autoplay></audio>
            </div>
            <script>
                // LiveKit client connection will be handled here
                let room = null;
                let localParticipant = null;
                
                async function connectToLiveKit(connectionInfo) {
                    try {
                        const info = JSON.parse(connectionInfo);
                        // Client-side LiveKit connection logic
                        console.log('Connecting to LiveKit:', info);
                        
                        // In production, implement proper LiveKit client SDK connection
                        // This is a placeholder for the actual implementation
                        
                    } catch (error) {
                        console.error('LiveKit connection error:', error);
                    }
                }
                
                // Listen for connection info updates
                window.addEventListener('message', (event) => {
                    if (event.data.type === 'livekit-connect') {
                        connectToLiveKit(event.data.connectionInfo);
                    }
                });
            </script>
            """,
            visible=True
        )
        
        # Event handlers
        def update_displays():
            """Update all display fields."""
            return (
                interface.get_client_info_display(),
                interface.get_mandate_history_display(),
                interface.get_current_call_step()
            )
        
        def fetch_data(user_id):
            """Fetch client data and update displays."""
            status = interface.update_client_data(user_id, preserve_workflow=False)
            client_info, mandate_history, call_step = update_displays()
            return status, interface.current_thread_id, client_info, mandate_history, call_step
        
        def refresh_data(user_id):
            """Refresh data while preserving workflow state."""
            status = interface.update_client_data(user_id, preserve_workflow=True)
            client_info, mandate_history, call_step = update_displays()
            console_capture.write("🔄 Data refreshed (workflow preserved)\n")
            return status, client_info, mandate_history, call_step
        
        async def connect_voice():
            """Connect to LiveKit server."""
            connection_info, status = await interface.connect_to_livekit()
            return connection_info, status
        
        async def disconnect_voice():
            """Disconnect from LiveKit server."""
            status = await interface.disconnect_from_livekit()
            return "", status
        
        def send_text_message(text, thread_id):
            """Send text message through workflow."""
            if not text.strip():
                return ""
            
            interface.process_text_input(text, thread_id)
            return ""  # Clear input
        
        def change_log_level(level):
            """Change logging level."""
            if interface.voice_handler:
                interface.voice_handler.set_log_level(level)
            return f"Log level set to: {level}"
        
        def clear_console():
            """Clear console output."""
            console_capture.clear()
            return "🖥️ Console cleared...\n"
        
        def clear_conversation_handler():
            """Clear conversation log."""
            console_capture.conversation_output = ""
            console_capture.write("💬 Conversation log cleared\n")
            return "💬 Conversation cleared...\n"
        
        # Wire up event handlers
        fetch_btn.click(
            fn=fetch_data,
            inputs=[user_id_input],
            outputs=[fetch_status, current_thread_id, client_info_display, mandate_history_display, call_step_display]
        )
        
        refresh_btn.click(
            fn=refresh_data,
            inputs=[user_id_input],
            outputs=[fetch_status, client_info_display, mandate_history_display, call_step_display]
        )
        
        connect_btn.click(
            fn=connect_voice,
            outputs=[livekit_connection, connection_status]
        )
        
        disconnect_btn.click(
            fn=disconnect_voice,
            outputs=[livekit_connection, connection_status]
        )
        
        send_btn.click(
            fn=send_text_message,
            inputs=[text_input, current_thread_id],
            outputs=[text_input]
        )
        
        text_input.submit(
            fn=send_text_message,
            inputs=[text_input, current_thread_id],
            outputs=[text_input]
        )
        
        log_level.change(
            fn=change_log_level,
            inputs=[log_level],
            outputs=[console_output]
        )
        
        clear_btn.click(fn=clear_console, outputs=[console_output])
        clear_conv_btn.click(fn=clear_conversation_handler, outputs=[conversation_output])
        
        # Periodic updates
        def update_logs():
            """Update console and conversation logs."""
            return console_capture.get_output(), console_capture.get_conversation_output()
        
        # Update logs every second
        demo.load(update_logs, outputs=[console_output, conversation_output], every=1)
        
        # Update call step every 2 seconds
        demo.load(
            lambda: interface.get_current_call_step(),
            outputs=[call_step_display],
            every=2
        )
        
        # Initialize console
        def initialize():
            console_capture.write(f"🚀 Voice Chat Test Console Initialized (LiveKit Edition)\n")
            console_capture.write(f"🕒 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            console_capture.write(f"🎙️ Using self-hosted LiveKit server\n")
            console_capture.write(f"📍 Live call step tracking enabled\n")
            console_capture.write(f"🔧 Tool message display enabled\n")
            console_capture.write(f"🧵 Session ID: {session_id}\n")
            console_capture.write(f"ℹ️ Instructions:\n")
            console_capture.write(f"   1. Enter User ID and click 'Fetch Data'\n")
            console_capture.write(f"   2. Click 'Connect Voice' to start voice chat\n")
            console_capture.write(f"   3. Speak naturally - LiveKit handles turn detection\n")
            console_capture.write(f"   4. Use text input for testing without voice\n")
            console_capture.write("-" * 50 + "\n")
            return console_capture.get_output(), console_capture.get_conversation_output()
        
        demo.load(initialize, outputs=[console_output, conversation_output])
    
    return demo


# Launch the application
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
