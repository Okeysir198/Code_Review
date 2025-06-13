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
from src.VAD_TurnDectection.hum_aware_VAD import HumAwareVADModel

# Configuration
load_dotenv(find_dotenv())
setup_logging()
logger = logging.getLogger(__name__)

TOKEN = os.environ.get("HF_TOKEN")

STT_CONFIG = {
    "model_name": "nvidia/parakeet-tdt-0.6b-v2",
    # "model_name": "whisper-large-v3-turbo",

    # Whisper configuration (alternative),
    "whisper-large-v3-turbo": {
        "checkpoint": "whisper-large-v3-turbo",
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

# Global STT model
stt_model = None

async def get_credentials():
    """Get Cloudflare TURN credentials for WebRTC"""
    return await get_cloudflare_turn_credentials_async(hf_token=TOKEN)

def initialize_stt_model():
    """Initialize and warm up the STT model"""
    logger.info("Initializing STT model...")
    model = create_stt_model(STT_CONFIG)
    
    # Warm up with dummy input
    logger.info("Warming up STT model")
    warmup_audio = np.zeros((16000,), dtype=np.float32)
    input_file = audio_to_file((16000, warmup_audio))
    model.transcribe(input_file)
    logger.info("Model warmup complete")
    
    return model

async def process_audio_input(audio_input: tuple[int, np.ndarray], transcript=""):
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

    # Get transcription
    try:
        transcription_result = stt_model.transcribe(output_file)
        new_transcription = transcription_result.get('text', '').strip()
        combined_transcription = transcript + new_transcription + "\n" if new_transcription else transcript
        logger.info(f"Transcription: '{new_transcription}'")
    except Exception as e:
        logger.error(f"STT transcription failed: {e}")
        combined_transcription = transcript
    
    # Yield processed audio
    yield (sample_rate, processed_audio)
    
    # Yield additional outputs
    yield AdditionalOutputs(input_file, output_file, combined_transcription)

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
                process_audio_input,
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
    stt_model = initialize_stt_model()
    
    demo = create_gradio_interface()
    demo.launch(
        share=False, 
        server_port=7860,
        server_name="0.0.0.0"
    )