"""
Real-time Audio Processing Application 
Uses FastRTC for WebRTC streaming 
"""

import os
import logging
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
from src.STT import create_stt_model
from src.TTS import create_tts_model

from src.VAD_TurnDectection.hum_aware_VAD import HumAwareVADModel
from src.Agents.graph_react_agent import react_agent_graph

# Configuration
load_dotenv(find_dotenv())
setup_logging()
logger = logging.getLogger(__name__)

TOKEN = os.environ.get("HF_TOKEN")

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

TTS_CONFIG = {
        "model_name": "kokorov2",  # Main selector for which TTS model to use
        
        # Enhanced Kokoro V2 settings
        # https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
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
    """Class to encapsulate STT model and processing logic"""
    
    def __init__(self):
        self.stt_model = None
        self.tts_model = None
        self.LLM = react_agent_graph
        self.initialize_stt_model()
        self.initialize_tts_model()
    
    def initialize_stt_model(self):
        """Initialize and warm up the STT model"""
        try:
            logger.info("Initializing STT model...")
            self.stt_model = create_stt_model(STT_CONFIG)
            
            # Warm up with dummy input
            logger.info("Warming up STT model")
            warmup_audio = np.zeros((16000,), dtype=np.float32)
            input_file = audio_to_file((16000, warmup_audio))
            self.stt_model.transcribe(input_file)
            logger.info("Model warmup complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize STT model: {e}")
            self.stt_model = None

    def initialize_tts_model(self):
        """Initialize the TTS model"""
        try:
            logger.info("Initializing TTS model...")
            self.tts_model = create_tts_model(TTS_CONFIG)
            logger.info("TTS model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}")
            self.tts_model = None
    
    def process_audio_input(self, audio_input: tuple[int, np.ndarray], transcript=""):
        """Process audio input and return transcription"""
        sample_rate, audio_array = audio_input
        logger.info(f"Processing audio - SR: {sample_rate}Hz, Shape: {audio_array.shape}")
        
        # Process audio (add your custom processing here if needed)
        processed_audio = audio_array.copy()

        # Prepare files
        input_audio_int16 = audio_to_int16(audio_array)
        processed_audio_int16 = audio_to_int16(processed_audio)
        
        input_file = audio_to_file((sample_rate, input_audio_int16))
        output_file = audio_to_file((sample_rate, processed_audio_int16))



        
        
        try:
            # Get transcription
            transcription_result = self.stt_model.transcribe(output_file)
            new_transcription = transcription_result.get('text', '').strip()
                    
            combined_transcription = transcript + new_transcription + "\n" if new_transcription else transcript
            logger.info(f"Transcription: '{new_transcription}'")
        
        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            combined_transcription = transcript + "[Transcription Error]\n"
        
        reponse = self.LLM.invoke(transcription_result)['messages'][-1].content
        logger.info(f"AI Respone: '{reponse}'")
        for audio_chunk in self.tts_model.stream_text_to_speech(reponse):
            yield audio_chunk

        
        
        # Yield additional outputs
        yield AdditionalOutputs(input_file, output_file, combined_transcription)

# Global audio processor instance
audio_processor = AudioProcessor()

async def get_credentials():
    """Get Cloudflare TURN credentials for WebRTC"""
    return await get_cloudflare_turn_credentials_async(hf_token=TOKEN)

def create_gradio_interface():
    """Create the Gradio interface"""
    with gr.Blocks(title="Real-time Audio Processing") as demo:
        gr.HTML("<h1 style='text-align: center'>🎤 Real-time Audio Processing with STT</h1>")
        
        with gr.Row():
            # Audio streaming
            with gr.Column():
                gr.Markdown("### Live Audio Stream")
                audio_stream = WebRTC(
                    label="Record & Process Audio",
                    mode="send-receive",
                    modality="audio",
                    track_constraints=AUDIO_CONSTRAINTS,
                    rtc_configuration=get_credentials,
                    min_width=80,
                )
            
            # Results
            with gr.Column():
                gr.Markdown("### Results")
                transcript = gr.Textbox(
                    label="Transcription",
                    placeholder="Transcribed text will appear here...",
                    lines=5,
                )
                
                gr.Markdown("### Audio Files")
                input_player = gr.Audio(
                    label="Input Audio", 
                    interactive=True,
                    format="wav",  
                    show_download_button=True
                )
                output_player = gr.Audio(
                    label="Processed Audio", 
                    interactive=True,
                    format="wav",  
                    show_download_button=True
                )
        
        # Set up audio streaming with VAD
        hum_vad_model = HumAwareVADModel()
        audio_stream.stream(
            fn=ReplyOnPause(
                audio_processor.process_audio_input,  # Use the class method
                input_sample_rate=16000,
                algo_options=ALGO_OPTIONS,
                model_options=VAD_OPTIONS,
                model=hum_vad_model,  # Using hum-aware VAD
                can_interrupt=False,
            ),
            inputs=[audio_stream, transcript],
            outputs=[audio_stream],
            time_limit=300,
            concurrency_limit=5,
        )

        # Handle additional outputs
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
    
    return demo

if __name__ == "__main__":
    print("Starting Real-time Audio Processing Application...")
    
    # Check if audio processor initialized successfully
    if audio_processor.stt_model is None:
        print("WARNING: STT model failed to initialize. Check logs for details.")
    else:
        print("STT model initialized successfully.")
    
    demo = create_gradio_interface()
    demo.launch(
        share=False, 
        server_port=7860,
        server_name="0.0.0.0"
    )