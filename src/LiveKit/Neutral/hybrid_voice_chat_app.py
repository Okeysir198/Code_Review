"""
Hybrid Voice Chat Application
FastRTC + LiveKit Audio Processing with Gradio Interface
Maintains existing UI while adding LiveKit audio features
"""

import gradio as gr
import asyncio
import numpy as np
import logging
import json
import base64
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import aiohttp

from voice_handler import HybridVoiceHandler, HybridAudioConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridGradioInterface:
    """Gradio interface for hybrid FastRTC + LiveKit voice chat"""
    
    def __init__(self):
        self.handler: Optional[HybridVoiceHandler] = None
        self.is_connected = False
        self.conversation_history = []
        self.peer_connection = None
        
        # WebRTC signaling state
        self.signaling_state = "disconnected"
        
        # Audio configuration
        self.audio_config = HybridAudioConfig()
        
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface maintaining original design"""
        
        with gr.Blocks(
            title="Voice Chat Assistant - Hybrid Mode",
            theme=gr.themes.Soft(),
            js=self._get_webrtc_javascript()
        ) as interface:
            
            gr.Markdown("# 🎙️ Voice Chat Assistant")
            gr.Markdown("FastRTC Transport + LiveKit Audio Processing")
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Chat display area
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=400,
                        elem_id="chatbot"
                    )
                    
                    # Hidden elements for WebRTC signaling
                    offer_data = gr.Textbox(
                        visible=False,
                        elem_id="offer_data"
                    )
                    answer_data = gr.Textbox(
                        visible=False,
                        elem_id="answer_data"
                    )
                    
                    # Control buttons
                    with gr.Row():
                        connect_btn = gr.Button(
                            "🔌 Connect",
                            variant="primary",
                            elem_id="connect_btn"
                        )
                        call_btn = gr.Button(
                            "📞 Start Call",
                            variant="secondary",
                            elem_id="call_btn",
                            interactive=False
                        )
                        end_btn = gr.Button(
                            "📴 End Call",
                            variant="stop",
                            elem_id="end_btn",
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
                    
                    # Audio visualization (placeholder)
                    audio_viz = gr.HTML(
                        value='<div id="audio-visualizer" style="height: 60px; background: #f0f0f0; border-radius: 8px; display: flex; align-items: center; justify-content: center;">Audio Visualizer</div>'
                    )
                
                with gr.Column(scale=1):
                    # Settings panel
                    gr.Markdown("### ⚙️ Settings")
                    
                    with gr.Accordion("🎚️ LiveKit Audio Processing", open=True):
                        use_livekit_vad = gr.Checkbox(
                            label="LiveKit VAD (Turn Detection)",
                            value=True,
                            elem_id="use_livekit_vad"
                        )
                        vad_sensitivity = gr.Slider(
                            label="VAD Sensitivity",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            elem_id="vad_sensitivity",
                            info="Higher = more sensitive"
                        )
                        noise_cancel = gr.Checkbox(
                            label="LiveKit Noise Cancellation",
                            value=True,
                            elem_id="noise_cancel"
                        )
                        noise_strength = gr.Slider(
                            label="Noise Reduction Strength",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.8,
                            step=0.1,
                            elem_id="noise_strength"
                        )
                        echo_cancel = gr.Checkbox(
                            label="Echo Cancellation",
                            value=True,
                            elem_id="echo_cancel"
                        )
                    
                    with gr.Accordion("🎤 Model Selection", open=True):
                        stt_provider = gr.Dropdown(
                            label="STT Provider",
                            choices=["deepgram", "google", "whisper", "azure"],
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
                    
                    with gr.Accordion("📊 Audio Stats", open=False):
                        audio_stats = gr.JSON(
                            label="Real-time Stats",
                            value={
                                "vad_active": False,
                                "noise_level": 0,
                                "speech_probability": 0,
                                "buffer_size": 0
                            },
                            elem_id="audio_stats"
                        )
            
            # Event handlers
            connect_btn.click(
                fn=self.initialize_handler,
                outputs=[status_text, connect_btn, call_btn]
            )
            
            call_btn.click(
                fn=self.start_call,
                outputs=[status_text, call_btn, end_btn, offer_data]
            )
            
            end_btn.click(
                fn=self.end_call,
                outputs=[status_text, call_btn, end_btn]
            )
            
            clear_btn.click(
                fn=self.clear_conversation,
                outputs=[chatbot]
            )
            
            # WebRTC answer handler
            answer_data.change(
                fn=self.handle_answer,
                inputs=[answer_data],
                outputs=[status_text]
            )
            
            # Audio settings handlers
            use_livekit_vad.change(
                fn=lambda x: self.update_audio_settings(use_livekit_vad=x),
                inputs=[use_livekit_vad]
            )
            
            vad_sensitivity.change(
                fn=lambda x: self.update_audio_settings(vad_sensitivity=x),
                inputs=[vad_sensitivity]
            )
            
            noise_cancel.change(
                fn=lambda x: self.update_audio_settings(use_noise_cancellation=x),
                inputs=[noise_cancel]
            )
            
            noise_strength.change(
                fn=lambda x: self.update_audio_settings(noise_reduction_strength=x),
                inputs=[noise_strength]
            )
            
            echo_cancel.change(
                fn=lambda x: self.update_audio_settings(use_echo_cancellation=x),
                inputs=[echo_cancel]
            )
            
            # Model selection handlers
            stt_provider.change(
                fn=self.switch_stt_provider,
                inputs=[stt_provider]
            )
            
            tts_provider.change(
                fn=self.switch_tts_provider,
                inputs=[tts_provider, tts_voice]
            )
            
            # Periodic stats update
            interface.load(
                fn=self.get_audio_stats,
                outputs=[audio_stats],
                every=1  # Update every second
            )
            
            # Custom CSS
            interface.css = """
                #chatbot {
                    border-radius: 10px;
                    border: 1px solid #e0e0e0;
                }
                .gradio-button {
                    border-radius: 6px;
                    font-weight: 500;
                }
                #connect_btn {
                    background-color: #4CAF50;
                }
                #call_btn {
                    background-color: #2196F3;
                }
                #end_btn {
                    background-color: #f44336;
                }
                #clear_btn {
                    background-color: #ff9800;
                }
                #status {
                    font-family: monospace;
                    font-size: 12px;
                }
                #audio-visualizer {
                    margin-top: 10px;
                    margin-bottom: 10px;
                }
                .audio-processing-label {
                    color: #673ab7;
                    font-weight: bold;
                }
            """
        
        return interface
    
    def _get_webrtc_javascript(self) -> str:
        """JavaScript for WebRTC handling"""
        return """
        async function() {
            // WebRTC configuration
            const configuration = {
                iceServers: [{urls: 'stun:stun.l.google.com:19302'}]
            };
            
            let pc = null;
            let localStream = null;
            
            // Initialize WebRTC when offer is available
            window.initializeWebRTC = async function(offerData) {
                try {
                    // Create peer connection
                    pc = new RTCPeerConnection(configuration);
                    
                    // Get user media
                    localStream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            echoCancellation: true,
                            noiseSuppression: true,
                            sampleRate: 16000
                        },
                        video: false
                    });
                    
                    // Add tracks to peer connection
                    localStream.getTracks().forEach(track => {
                        pc.addTrack(track, localStream);
                    });
                    
                    // Handle incoming tracks
                    pc.ontrack = (event) => {
                        console.log('Received remote track');
                        const audio = new Audio();
                        audio.srcObject = event.streams[0];
                        audio.play();
                    };
                    
                    // Set remote description (offer)
                    const offer = JSON.parse(offerData);
                    await pc.setRemoteDescription(offer);
                    
                    // Create answer
                    const answer = await pc.createAnswer();
                    await pc.setLocalDescription(answer);
                    
                    // Send answer back to Python
                    const answerData = document.getElementById('answer_data');
                    answerData.value = JSON.stringify({
                        sdp: answer.sdp,
                        type: answer.type
                    });
                    answerData.dispatchEvent(new Event('input', {bubbles: true}));
                    
                } catch (error) {
                    console.error('WebRTC initialization error:', error);
                }
            };
            
            // Clean up WebRTC
            window.cleanupWebRTC = function() {
                if (localStream) {
                    localStream.getTracks().forEach(track => track.stop());
                }
                if (pc) {
                    pc.close();
                }
            };
            
            // Audio visualization (simple example)
            window.startAudioVisualization = function() {
                if (!localStream) return;
                
                const audioContext = new AudioContext();
                const analyser = audioContext.createAnalyser();
                const source = audioContext.createMediaStreamSource(localStream);
                source.connect(analyser);
                
                const canvas = document.createElement('canvas');
                canvas.width = 300;
                canvas.height = 60;
                document.getElementById('audio-visualizer').innerHTML = '';
                document.getElementById('audio-visualizer').appendChild(canvas);
                
                const ctx = canvas.getContext('2d');
                const bufferLength = analyser.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);
                
                function draw() {
                    requestAnimationFrame(draw);
                    analyser.getByteFrequencyData(dataArray);
                    
                    ctx.fillStyle = '#f0f0f0';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    const barWidth = (canvas.width / bufferLength) * 2.5;
                    let x = 0;
                    
                    for (let i = 0; i < bufferLength; i++) {
                        const barHeight = (dataArray[i] / 255) * canvas.height;
                        ctx.fillStyle = '#2196F3';
                        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                        x += barWidth + 1;
                    }
                }
                
                draw();
            };
        }
        """
    
    async def initialize_handler(self) -> Tuple[str, str, bool]:
        """Initialize the hybrid handler"""
        try:
            self.handler = HybridVoiceHandler(config=self.audio_config)
            await self.handler.initialize()
            
            self.is_connected = True
            return "✅ Handler initialized", "🔌 Reconnect", True
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return f"❌ Error: {str(e)}", "🔌 Connect", False
    
    async def start_call(self) -> Tuple[str, bool, bool, str]:
        """Start WebRTC call"""
        try:
            if not self.handler:
                return "❌ Handler not initialized", True, False, ""
            
            # Create offer
            offer = await self.handler.create_offer()
            offer_json = json.dumps(offer)
            
            return "📞 Calling... Waiting for answer", False, True, offer_json
            
        except Exception as e:
            logger.error(f"Call error: {e}")
            return f"❌ Error: {str(e)}", True, False, ""
    
    async def handle_answer(self, answer_json: str) -> str:
        """Handle WebRTC answer"""
        try:
            if not answer_json or not self.handler:
                return self.signaling_state
            
            answer = json.loads(answer_json)
            await self.handler.handle_answer(answer)
            
            return "✅ Connected - Voice chat active"
            
        except Exception as e:
            logger.error(f"Answer handling error: {e}")
            return f"❌ Error: {str(e)}"
    
    async def end_call(self) -> Tuple[str, bool, bool]:
        """End the call"""
        try:
            if self.handler:
                await self.handler.close()
            
            return "📴 Call ended", True, False
            
        except Exception as e:
            logger.error(f"End call error: {e}")
            return f"❌ Error: {str(e)}", True, False
    
    def update_audio_settings(self, **kwargs):
        """Update audio processing settings"""
        # Update config
        for key, value in kwargs.items():
            if hasattr(self.audio_config, key):
                setattr(self.audio_config, key, value)
        
        # Apply to handler if connected
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
    
    def get_audio_stats(self) -> Dict[str, Any]:
        """Get real-time audio statistics"""
        if self.handler and self.handler.audio_processor:
            return {
                "vad_active": self.handler.audio_processor.is_speaking,
                "noise_level": 0,  # Would need to implement
                "speech_probability": 0,  # Would need to implement
                "buffer_size": len(self.handler.audio_buffer),
                "livekit_features": {
                    "vad_enabled": self.audio_config.use_livekit_vad,
                    "noise_cancel": self.audio_config.use_noise_cancellation,
                    "echo_cancel": self.audio_config.use_echo_cancellation
                }
            }
        return {
            "vad_active": False,
            "noise_level": 0,
            "speech_probability": 0,
            "buffer_size": 0,
            "livekit_features": {}
        }


def main():
    """Main entry point"""
    # Create interface
    app = HybridGradioInterface()
    interface = app.create_interface()
    
    # Launch with FastRTC-compatible settings
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        ssl_verify=False  # For local WebRTC testing
    )


if __name__ == "__main__":
    # Run with asyncio support
    import nest_asyncio
    nest_asyncio.apply()
    
    main()
