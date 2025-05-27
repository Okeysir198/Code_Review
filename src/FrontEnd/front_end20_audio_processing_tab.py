#./src/FrontEnd/interfaces/front_end_0106_audio_processing_tab.py
"""
Audio processing tab for the Call Center AI Agent interface.
"""

import logging
from typing import Dict, Any, List
import traceback
import os

import gradio as gr
from src.STT import create_stt_model
from app_config import CONFIG

logger = logging.getLogger(__name__)


def get_audio_samples() -> List[str]:
    """Find audio examples in specified folder for sample demonstrations."""
    folder_path = "audio_samples"

    if not os.path.exists(folder_path):
        return []
        
    try:
        return [
            os.path.join(folder_path, item) 
            for item in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, item))
        ]
    except Exception as e:
        logger.error(f"Audio samples loading error: {str(e)}\n{traceback.format_exc()}")
        return []

def create_audio_processing_tab(config: Dict[str, Any], visible: bool=True):
    """Build audio transcription tab with file upload and processing."""
    config = CONFIG.copy()['stt']
    stt_model = create_stt_model(config)
    sample_files= get_audio_samples()

    with gr.Tab("Audio Processing", visible=visible) as tab:
        gr.Markdown("## Transcribe audio files")
        gr.Markdown(
            "Transcribe audio files with the click of a button! "
            f"Demo uses the STT model [{config.get('model_name')}]."
        )
        
        
        with gr.Row():
            with gr.Column():
                # Input components
                audio_input = gr.Audio(
                    sources="upload", 
                    type="filepath", 
                    label="Transcribe Audio File"
                )
                
                task_selector = gr.Radio(
                    ["transcribe", "translate"], 
                    label="Task", 
                    value="transcribe"
                )
                
                transcribe_btn = gr.Button("Transcribe", variant="primary")
                
            with gr.Column():
                transcript_output = gr.Textbox(
                    label="Transcript",
                    lines=10,
                    max_lines=20
                )
        gr.Markdown("---")
        # Add examples if available
        if sample_files:
            gr.Examples(
                examples=sample_files,
                inputs=audio_input,
                cache_examples=False,
            )
        # transcribe
        def transcribe_handler(audio_input, task):
            return stt_model.transcribe(audio_input,task).get('text',"")
        
        # Connect transcription function
        transcribe_btn.click(
            fn=transcribe_handler,
            inputs=[audio_input, task_selector],
            outputs=transcript_output,
            queue=True
        )
        
        return tab
