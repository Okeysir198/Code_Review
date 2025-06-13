"""
Real-time Audio Processing Application with Simplified Voice Handler
Uses FastRTC for WebRTC streaming with any LangGraph workflow
"""

import os
import logging
import uuid
import numpy as np
import gradio as gr
from dotenv import find_dotenv, load_dotenv

# FastRTC imports
from fastrtc import (
    WebRTC, 
    ReplyOnPause, 
    AlgoOptions,
    SileroVadOptions,
    AdditionalOutputs, 
    get_cloudflare_turn_credentials_async,
    audio_to_int16, 
    audio_to_file
)

# Utils imports
from src.utils.logger_config import setup_logging
from src.VAD_TurnDectection.hum_aware_VAD import HumAwareVADModel
from src.Agents.graph_react_agent import react_agent_graph

# Import the simplified voice handler
from src.VoiceHandler.simple_voice_handler import SimpleVoiceHandler

# Configuration
load_dotenv(find_dotenv())
setup_logging()
logger = logging.getLogger(__name__)

TOKEN = os.environ.get("HF_TOKEN")

# STT Configuration
STT_CONFIG = {
    "model_name": "nvidia/parakeet-tdt-0.6b-v2",
    # "model_name": "openai/whisper-large-v3-turbo",

    # Whisper configuration (alternative),
    "openai/whisper-large-v3-turbo": {
        "checkpoint": "openai/whisper-large-v3-turbo",
        "model_folder_path": "/home/ct-admin/Documents/Langgraph/HF_models/",
        "batch_size": 4,
        "cuda_device_id": 1,
        "chunk_length_s": 30,
        "compute_type": "float16",
        "beam_size": 3
    },
    
    # NVIDIA Parakeet configuration
    "nvidia/parakeet-tdt-0.6b-v2": {
        "timestamp_prediction": True,
        "decoding_type": "tdt"
    }
}

# TTS Configuration
TTS_CONFIG = {
    "model_name": "kokorov2",  # Main selector for which TTS model to use
    
    # Enhanced Kokoro V2 settings
    "kokorov2": {
        "voice": "am_michael",  # af_bella, af_heart, am_fenrir, am_michael
        "speed": 1.3,
        "language": "a",  # 'a' for US English, 'b' for UK English
        "use_gpu": True,
        "fallback_to_cpu": True,
        "sample_rate": 24000,
        "preload_voices": ["af_heart"],  # Preload common voices
        "custom_pronunciations": {
            "kokoro": {"a": "kˈOkəɹO", "b": "kˈQkəɹQ"},
            "cartrack": {"a": "kˈɑɹtɹæk", "b": "kˈɑːtɹæk"}
        }
    },
}

# Audio constraints for WebRTC
AUDIO_CONSTRAINTS = {
    "noiseSuppression": {"exact": True},
    "autoGainControl": {"exact": True},
    "sampleRate": {"ideal": 16000},
    "channelCount": {"exact": 1},
    "googNoiseSuppression": {"exact": True},
    "googEchoCancellation": {"exact": True},
    "googHighpassFilter": {"exact": True}
}

# VAD Options
VAD_OPTIONS = SileroVadOptions(
    threshold=0.5,
    min_speech_duration_ms=250,
    max_speech_duration_s=30,
    min_silence_duration_ms=500,
    window_size_samples=1024,
    speech_pad_ms=50,
)

# Algorithm Options
ALGO_OPTIONS = AlgoOptions(
    audio_chunk_duration=0.6,
    started_talking_threshold=0.2,
    speech_threshold=0.1,
)

class AudioProcessor:
    """Simplified audio processor using the generic voice handler"""
    
    def __init__(self):
        """Initialize the voice handler with STT and TTS models"""
        self.voice_handler = SimpleVoiceHandler(
            stt_config=STT_CONFIG,
            tts_config=TTS_CONFIG,
            enable_stt=True,
            enable_tts=True
        )
        
        # Add message handler for console logging
        self.voice_handler.add_message_handler(self._log_message)
        
        # Current conversation state
        self.current_chatbot = []
        self.current_thread_id = str(uuid.uuid4())
        
        logger.info("AudioProcessor initialized with SimpleVoiceHandler")
    
    def _log_message(self, role: str, content: str):
        """Log messages to console with formatting"""
        timestamp = __import__('time').strftime("%H:%M:%S")
        print(f"[{timestamp}] {role.upper()}: {content}")
    
    def process_audio_input(self, audio_input: tuple[int, np.ndarray], transcript=""):
        """
        Process audio input using the simplified voice handler.
        This replaces the complex processing logic with a clean interface.
        """
        try:
            sample_rate, audio_array = audio_input
            logger.info(f"Processing audio - SR: {sample_rate}Hz, Shape: {audio_array.shape}")
            
            # Validate audio input
            if not audio_array.size:
                logger.warning("Empty audio array received")
                yield (sample_rate, np.array([0], dtype=np.int16))
                yield AdditionalOutputs(None, None, transcript)
                return
            
            # Create graph configuration with thread ID
            config = {
                "configurable": {
                    "thread_id": self.current_thread_id
                }
            }
            
            # Process audio through the voice handler
            final_audio = None
            final_chatbot = self.current_chatbot.copy()
            
            for result in self.voice_handler.process_audio_input(
                audio_input, 
                react_agent_graph, 
                config, 
                self.current_chatbot
            ):
                if isinstance(result, AdditionalOutputs):
                    # Update chatbot state
                    final_chatbot = result
                elif isinstance(result, tuple) and len(result) == 2:
                    # Audio output
                    final_audio = result
            
            # Update current state
            self.current_chatbot = final_chatbot
            
            # Prepare file outputs for UI
            input_audio_int16 = audio_to_int16(audio_array)
            input_file = audio_to_file((sample_rate, input_audio_int16))
            
            # Use the final audio or create silence
            if final_audio is not None:
                output_file = audio_to_file(final_audio)
                yield final_audio  # Yield the actual TTS audio
            else:
                output_file = audio_to_file((sample_rate, np.array([0], dtype=np.int16)))
                yield (sample_rate, np.array([0], dtype=np.int16))
            
            # Update transcript with latest conversation
            updated_transcript = self._format_transcript()
            
            # Yield additional outputs for UI
            yield AdditionalOutputs(input_file, output_file, updated_transcript)
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            # Fallback response
            yield (audio_input[0], np.array([0], dtype=np.int16))
            yield AdditionalOutputs(None, None, transcript + f"\n[Error: {e}]")
    
    def _format_transcript(self) -> str:
        """Format the current chatbot conversation as a transcript"""
        transcript_lines = []
        for msg in self.current_chatbot:
            role = msg.get("role", "unknown").title()
            content = msg.get("content", "")
            transcript_lines.append(f"{role}: {content}")
        
        return "\n".join(transcript_lines)
    
    def process_text_input(self, text_input: str, current_transcript: str = ""):
        """Process text input through the voice handler"""
        try:
            config = {
                "configurable": {
                    "thread_id": self.current_thread_id
                }
            }
            
            # Process text through voice handler
            final_audio = None
            final_chatbot = self.current_chatbot.copy()
            
            for result in self.voice_handler.process_text_input(
                text_input,
                react_agent_graph,
                config,
                self.current_chatbot
            ):
                audio_chunk, chatbot_update, _ = result
                if audio_chunk is not None:
                    final_audio = audio_chunk
                if chatbot_update:
                    final_chatbot = chatbot_update
            
            # Update current state
            self.current_chatbot = final_chatbot
            
            # Return updated transcript and audio
            updated_transcript = self._format_transcript()
            return updated_transcript, final_audio
            
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return current_transcript + f"\n[Error: {e}]", None
    
    def reset_conversation(self):
        """Reset the conversation state"""
        self.current_chatbot = []
        self.current_thread_id = str(uuid.uuid4())
        logger.info("Conversation reset")

# Global audio processor instance
audio_processor = AudioProcessor()

async def get_credentials():
    """Get Cloudflare TURN credentials for WebRTC"""
    return await get_cloudflare_turn_credentials_async(hf_token=TOKEN)

def create_gradio_interface():
    """Create the Gradio interface with enhanced functionality"""
    with gr.Blocks(title="Real-time Audio Processing") as demo:
        gr.HTML("<h1 style='text-align: center'>🎤 Real-time Audio Chat with React Agent</h1>")
        
        with gr.Row():
            # Left column - Audio streaming
            with gr.Column(scale=2):
                gr.Markdown("### 🎙️ Live Audio Stream")
                audio_stream = WebRTC(
                    label="Record & Process Audio",
                    mode="send-receive",
                    modality="audio",
                    track_constraints=AUDIO_CONSTRAINTS,
                    rtc_configuration=get_credentials,
                    min_width=80,
                )
                
                # Text input for manual messaging
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Type your message",
                        placeholder="Type here to chat with the agent...",
                        scale=4
                    )
                    send_btn = gr.Button("Send", scale=1)
                
                # Conversation controls
                with gr.Row():
                    reset_btn = gr.Button("🔄 Reset Conversation", variant="secondary")
            
            # Right column - Results and conversation
            with gr.Column(scale=3):
                gr.Markdown("### 💬 Conversation")
                transcript = gr.Textbox(
                    label="Live Transcript",
                    placeholder="Conversation will appear here...",
                    lines=15,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True
                )
                
                gr.Markdown("### 🔊 Audio Files")
                with gr.Row():
                    input_player = gr.Audio(
                        label="Your Audio", 
                        interactive=False,
                        format="wav",  
                        show_download_button=True,
                        scale=1
                    )
                    output_player = gr.Audio(
                        label="AI Response", 
                        interactive=False,
                        format="wav",  
                        show_download_button=True,
                        autoplay=True,
                        scale=1
                    )
        
        # Set up audio streaming with VAD
        hum_vad_model = HumAwareVADModel()
        audio_stream.stream(
            fn=ReplyOnPause(
                audio_processor.process_audio_input,
                input_sample_rate=16000,
                algo_options=ALGO_OPTIONS,
                model_options=VAD_OPTIONS,
                model=hum_vad_model,
                can_interrupt=False,
            ),
            inputs=[audio_stream, transcript],
            outputs=[audio_stream],
            time_limit=300,
            concurrency_limit=5,
        )

        # Handle additional outputs (audio files and transcript)
        audio_stream.on_additional_outputs(
            fn=lambda input_audio, output_audio, transcript_text: (
                input_audio, 
                output_audio, 
                transcript_text
            ),
            outputs=[input_player, output_player, transcript],
            queue=False, 
            show_progress="hidden"
        )
        
        # Text input handling
        def handle_text_input(text, current_transcript):
            if not text.strip():
                return current_transcript, "", None
            
            updated_transcript, audio_response = audio_processor.process_text_input(text, current_transcript)
            return updated_transcript, "", audio_response
        
        send_btn.click(
            handle_text_input,
            inputs=[text_input, transcript],
            outputs=[transcript, text_input, output_player]
        )
        
        text_input.submit(
            handle_text_input,
            inputs=[text_input, transcript],
            outputs=[transcript, text_input, output_player]
        )
        
        # Reset conversation
        def reset_conversation():
            audio_processor.reset_conversation()
            return ""
        
        reset_btn.click(
            reset_conversation,
            outputs=[transcript]
        )
    
    return demo

if __name__ == "__main__":
    print("Starting Real-time Audio Processing Application...")
    
    # Check if voice handler initialized successfully
    if audio_processor.voice_handler.stt_model is None:
        print("WARNING: STT model failed to initialize. Check logs for details.")
    else:
        print("✅ STT model initialized successfully.")
    
    if audio_processor.voice_handler.tts_model is None:
        print("WARNING: TTS model failed to initialize. Check logs for details.")
    else:
        print("✅ TTS model initialized successfully.")
    
    print("🚀 Launching Gradio interface...")
    demo = create_gradio_interface()
    demo.launch(
        share=False, 
        server_port=7860,
        server_name="0.0.0.0"
    )