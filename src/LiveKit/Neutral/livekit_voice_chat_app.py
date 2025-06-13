"""
LiveKit Voice Chat Application with Gradio Interface
Maintains existing UI design while using LiveKit backend
"""

import gradio as gr
import asyncio
import numpy as np
import logging
from typing import Optional, Generator, Tuple
import pyaudio
import wave
import io
import base64
from datetime import datetime

from voice_handler import LiveKitVoiceHandler, AudioConfig, create_langgraph_workflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveKitGradioInterface:
    """Gradio interface for LiveKit voice chat"""
    
    def __init__(self):
        self.handler: Optional[LiveKitVoiceHandler] = None
        self.is_connected = False
        self.is_recording = False
        self.audio_thread = None
        self.conversation_history = []
        
        # Audio settings (matching original design)
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        
        # PyAudio for local audio capture
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        
        # LiveKit settings
        self.livekit_url = "ws://localhost:7880"
        self.api_key = "devkey"
        self.api_secret = "secret"
        
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface matching original design"""
        
        with gr.Blocks(title="Voice Chat Assistant", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎙️ Voice Chat Assistant")
            gr.Markdown("Powered by LiveKit with LangGraph Integration")
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Chat display area
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=400,
                        elem_id="chatbot"
                    )
                    
                    # Voice input controls
                    with gr.Row():
                        audio_input = gr.Audio(
                            label="Voice Input",
                            source="microphone",
                            type="numpy",
                            streaming=True,
                            elem_id="audio_input"
                        )
                        
                    # Control buttons
                    with gr.Row():
                        connect_btn = gr.Button(
                            "🔌 Connect", 
                            variant="primary",
                            elem_id="connect_btn"
                        )
                        record_btn = gr.Button(
                            "🎤 Start Recording",
                            variant="secondary",
                            elem_id="record_btn",
                            interactive=False
                        )
                        clear_btn = gr.Button(
                            "🗑️ Clear",
                            elem_id="clear_btn"
                        )
                    
                    # Status display
                    status_text = gr.Textbox(
                        label="Status",
                        value="Disconnected",
                        interactive=False,
                        elem_id="status"
                    )
                
                with gr.Column(scale=1):
                    # Settings panel
                    gr.Markdown("### ⚙️ Settings")
                    
                    with gr.Accordion("Audio Settings", open=True):
                        noise_cancel = gr.Checkbox(
                            label="Noise Cancellation",
                            value=True,
                            elem_id="noise_cancel"
                        )
                        echo_cancel = gr.Checkbox(
                            label="Echo Cancellation",
                            value=True,
                            elem_id="echo_cancel"
                        )
                        vad_sensitivity = gr.Slider(
                            label="VAD Sensitivity",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            elem_id="vad_sensitivity"
                        )
                    
                    with gr.Accordion("Model Selection", open=True):
                        stt_provider = gr.Dropdown(
                            label="STT Provider",
                            choices=["deepgram", "google", "azure", "whisper"],
                            value="deepgram",
                            elem_id="stt_provider"
                        )
                        tts_provider = gr.Dropdown(
                            label="TTS Provider",
                            choices=["elevenlabs", "google", "azure", "cartesia"],
                            value="elevenlabs",
                            elem_id="tts_provider"
                        )
                        tts_voice = gr.Dropdown(
                            label="Voice",
                            choices=["Rachel", "Josh", "Emily", "Adam"],
                            value="Rachel",
                            elem_id="tts_voice"
                        )
                    
                    with gr.Accordion("Connection", open=False):
                        room_name = gr.Textbox(
                            label="Room Name",
                            value="voice-chat-room",
                            elem_id="room_name"
                        )
                        participant_id = gr.Textbox(
                            label="Participant ID",
                            value="user-" + str(datetime.now().timestamp())[:8],
                            elem_id="participant_id"
                        )
            
            # Event handlers
            connect_btn.click(
                fn=self.toggle_connection,
                inputs=[room_name, participant_id],
                outputs=[status_text, connect_btn, record_btn]
            )
            
            record_btn.click(
                fn=self.toggle_recording,
                outputs=[record_btn, status_text]
            )
            
            clear_btn.click(
                fn=self.clear_conversation,
                outputs=[chatbot]
            )
            
            # Audio streaming
            audio_input.stream(
                fn=self.process_audio_stream,
                inputs=[audio_input],
                outputs=[chatbot],
                show_progress=False
            )
            
            # Settings updates
            noise_cancel.change(
                fn=lambda x: self.update_audio_settings(noise_cancellation=x),
                inputs=[noise_cancel]
            )
            
            echo_cancel.change(
                fn=lambda x: self.update_audio_settings(echo_cancellation=x),
                inputs=[echo_cancel]
            )
            
            vad_sensitivity.change(
                fn=lambda x: self.update_audio_settings(vad_sensitivity=x),
                inputs=[vad_sensitivity]
            )
            
            stt_provider.change(
                fn=self.switch_stt_provider,
                inputs=[stt_provider]
            )
            
            tts_provider.change(
                fn=self.switch_tts_provider,
                inputs=[tts_provider, tts_voice]
            )
            
            # Custom CSS for original design
            interface.css = """
                #chatbot {
                    border-radius: 10px;
                    border: 1px solid #e0e0e0;
                }
                #audio_input {
                    border-radius: 8px;
                }
                .gradio-button {
                    border-radius: 6px;
                    font-weight: 500;
                }
                #connect_btn {
                    background-color: #4CAF50;
                }
                #record_btn {
                    background-color: #2196F3;
                }
                #clear_btn {
                    background-color: #f44336;
                }
                #status {
                    font-family: monospace;
                    font-size: 12px;
                }
            """
        
        return interface
    
    async def toggle_connection(self, room_name: str, participant_id: str) -> Tuple[str, str, bool]:
        """Toggle LiveKit connection"""
        try:
            if not self.is_connected:
                # Create LangGraph workflow
                workflow = create_langgraph_workflow()
                
                # Initialize handler
                self.handler = LiveKitVoiceHandler(
                    room_name=room_name,
                    participant_identity=participant_id,
                    langgraph_workflow=workflow
                )
                
                # Connect to LiveKit
                await self.handler.connect(
                    self.livekit_url,
                    self.api_key,
                    self.api_secret
                )
                
                self.is_connected = True
                return "✅ Connected to LiveKit", "🔌 Disconnect", True
            else:
                # Disconnect
                if self.handler:
                    await self.handler.disconnect()
                    self.handler = None
                
                self.is_connected = False
                return "❌ Disconnected", "🔌 Connect", False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return f"❌ Error: {str(e)}", "🔌 Connect", False
    
    def toggle_recording(self) -> Tuple[str, str]:
        """Toggle audio recording"""
        try:
            if not self.is_recording:
                self.start_recording()
                return "🔴 Stop Recording", "🔴 Recording..."
            else:
                self.stop_recording()
                return "🎤 Start Recording", "✅ Ready"
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return "🎤 Start Recording", f"❌ Error: {str(e)}"
    
    def start_recording(self):
        """Start audio recording"""
        if not self.is_connected or not self.handler:
            raise Exception("Not connected to LiveKit")
        
        self.is_recording = True
        
        # Open PyAudio stream
        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        self.stream.start_stream()
        logger.info("Started recording")
    
    def stop_recording(self):
        """Stop audio recording"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        logger.info("Stopped recording")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio streaming"""
        if self.is_recording and self.handler:
            # Send audio to LiveKit asynchronously
            asyncio.create_task(self._send_audio_chunk(in_data))
        
        return (in_data, pyaudio.paContinue)
    
    async def _send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk to LiveKit"""
        if self.handler:
            await self.handler.process_audio_stream(
                self._audio_generator([audio_data])
            )
    
    async def _audio_generator(self, chunks):
        """Convert audio chunks to async generator"""
        for chunk in chunks:
            yield chunk
    
    def process_audio_stream(self, audio_data: Optional[np.ndarray]) -> list:
        """Process streaming audio from Gradio"""
        if audio_data is None or not self.is_connected:
            return self.conversation_history
        
        # Convert numpy array to bytes
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        audio_bytes = audio_data.tobytes()
        
        # Send to LiveKit
        if self.handler:
            asyncio.create_task(
                self.handler.process_audio_stream(
                    self._audio_generator([audio_bytes])
                )
            )
        
        return self.conversation_history
    
    def update_audio_settings(self, **kwargs):
        """Update audio processing settings"""
        if self.handler:
            asyncio.create_task(
                self.handler.update_audio_settings(**kwargs)
            )
    
    def switch_stt_provider(self, provider: str):
        """Switch STT provider"""
        if self.handler:
            asyncio.create_task(
                self.handler.switch_stt_provider(provider)
            )
    
    def switch_tts_provider(self, provider: str, voice: str):
        """Switch TTS provider and voice"""
        if self.handler:
            # Map voice names to provider-specific IDs
            voice_mapping = {
                "elevenlabs": {
                    "Rachel": "EXAVITQu4vr4xnSDxMaL",
                    "Josh": "TxGEqnHWrfWFTfGW9XjX",
                    "Emily": "LcfcDJNUP1GQjkzn1xUU",
                    "Adam": "pNInz6obpgDQGcFmaJgB"
                }
            }
            
            voice_id = voice_mapping.get(provider, {}).get(voice, voice)
            
            asyncio.create_task(
                self.handler.switch_tts_provider(provider, voice_id=voice_id)
            )
    
    def clear_conversation(self) -> list:
        """Clear conversation history"""
        self.conversation_history = []
        return []
    
    def __del__(self):
        """Cleanup resources"""
        if self.stream:
            self.stream.close()
        if self.pyaudio:
            self.pyaudio.terminate()


def main():
    """Main entry point"""
    # Create interface
    app = LiveKitGradioInterface()
    interface = app.create_interface()
    
    # Launch
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )


if __name__ == "__main__":
    main()
