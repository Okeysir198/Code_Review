"""
Real-time Audio Processing Application with STT/TTS
Uses FastRTC for WebRTC streaming and LangChain for AI responses
"""

import os
import logging
import uuid
from typing import Generator, Tuple
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
    get_stt_model,
    audio_to_file
)

# Utils imports
from src.utils.logger_config import setup_logging
from src.STT import create_stt_model
from src.TTS import create_tts_model
from src.VAD_TurnDectection.hum_aware_VAD import HumAwareVADModel

# LangChain imports
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import HumanMessage

# Configuration
load_dotenv(find_dotenv())
setup_logging()
logger = logging.getLogger(__name__)

# Environment variables
TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    logger.warning("HF_TOKEN not found in environment variables")

# Model configurations
STT_CONFIG = {
    "model_name": "openai/whisper-large-v3-turbo",
    
    # Whisper configuration (alternative)
    "openai/whisper-large-v3-turbo": {
        "checkpoint": "openai/whisper-large-v3-turbo",
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
    "model_name": "kokorov2",
    
    # Enhanced Kokoro V2 settings
    "kokorov2": {
        "voice": "am_michael",  # af_bella, af_heart, am_fenrir, am_michael
        "speed": 1.3,
        "language": "a",  # 'a' for US English, 'b' for UK English
        "use_gpu": True,
        "fallback_to_cpu": True,
        "sample_rate": 24000,
        "preload_voices": ["af_heart"],
        "custom_pronunciations": {
            "kokoro": {"a": "kˈOkəɹO", "b": "kˈQkəɹQ"},
            "cartrack": {"a": "kˈɑɹtɹæk", "b": "kˈɑːtɹæk"}
        }
    }
}

# WebRTC audio constraints
AUDIO_CONSTRAINTS = {
    "noiseSuppression": {"exact": True},
    "autoGainControl": {"exact": True},
    "sampleRate": {"ideal": 16000},
    "channelCount": {"exact": 1},
    "googNoiseSuppression": {"exact": True},
    "googEchoCancellation": {"exact": True},
    "googHighpassFilter": {"exact": True}
}

# Voice Activity Detection options
VAD_OPTIONS = SileroVadOptions(
    threshold=0.5,
    min_speech_duration_ms=250,
    max_speech_duration_s=30,
    min_silence_duration_ms=500,
    window_size_samples=1024,
    speech_pad_ms=400,
)

# Algorithm options for audio processing
ALGO_OPTIONS = AlgoOptions(
    audio_chunk_duration=0.6,
    started_talking_threshold=0.3,
    speech_threshold=0.2,
)

class AudioProcessor:
    """
    Main class for processing audio input/output with STT, TTS, and AI conversation
    """
    
    def __init__(self):
        """Initialize all models and components"""
        self.stt_model = None
        self.tts_model = None
        self.llm_model = None
        self.agent: CompiledGraph = None
        self.checkpointer = MemorySaver()
        self.current_thread_id = str(uuid.uuid4())
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all AI models"""
        try:
            self._initialize_llm()
            self._initialize_stt_model()
            self._initialize_tts_model()
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
    
    def _initialize_llm(self):
        """Initialize the language model and agent"""
        try:
            logger.info("Initializing LLM and agent...")
            self.llm_model = init_chat_model(
                "ollama:qwen2.5:14b-instruct", 
                # "anthropic:claude-3-5-sonnet-20241022",
                temperature=0.1
            )
            self.llm_model.invoke("") #Please keep responses concise (30 words or less). 
            self.agent = create_react_agent(
                model=self.llm_model,
                tools=[],
                prompt=(
                    "You are a debt collection specialist from Cartrack Account Department. "
                    "Your name is AI Agent. "
                    "Be professional and helpful."
                ),
                checkpointer=self.checkpointer
            )
            logger.info("LLM and agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm_model = None
            self.agent = None
    
    def _initialize_stt_model(self):
        """Initialize and warm up the Speech-to-Text model"""
        try:
            logger.info("Initializing STT model...")
            self.stt_model = create_stt_model(STT_CONFIG)
            # self.stt_model = get_stt_model(model="moonshine/base")
            
            # Warm up with dummy input
            logger.info("Warming up STT model...")
            warmup_audio = np.zeros((16000,), dtype=np.float32)
            # input_file = audio_to_file((16000, warmup_audio))
            self.stt_model.transcribe((16000, warmup_audio))
            # self.stt_model.stt((16000, warmup_audio))
            logger.info("STT model warmup complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize STT model: {e}")
            self.stt_model = None
    
    def _initialize_tts_model(self):
        """Initialize the Text-to-Speech model"""
        try:
            logger.info("Initializing TTS model...")
            self.tts_model = create_tts_model(TTS_CONFIG)
            logger.info("TTS model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}")
            self.tts_model = None
    
    def start_new_conversation(self) -> str:
        """
        Start a new conversation with a fresh thread ID
        
        Returns:
            str: New thread ID
        """
        self.current_thread_id = str(uuid.uuid4())
        logger.info(f"Started new conversation with thread ID: {self.current_thread_id}")
        return self.current_thread_id
    
    def get_current_thread_id(self) -> str:
        """Get the current conversation thread ID"""
        return self.current_thread_id
    
    def process_audio_input(
        self, 
        audio_input: Tuple[int, np.ndarray], 
        transcript: str = "", 
        thread_id: str = None
    ) -> Generator:
        """
        Process audio input and return AI response
        
        Args:
            audio_input: Tuple of (sample_rate, audio_array)
            transcript: Current transcript text
            thread_id: Conversation thread ID (uses current if None)
            
        Yields:
            Audio chunks for TTS output and additional outputs
        """
        if thread_id is None:
            thread_id = self.current_thread_id
            
        sample_rate, audio_array = audio_input
        logger.info(f"Processing audio - SR: {sample_rate}Hz, Shape: {audio_array.shape}, Type: {audio_array.dtype}, Max. Value: {np.max(audio_array[0,:])}")
        
        # Check if models are available
        if not self.stt_model:
            logger.error("STT model not available")
            yield AdditionalOutputs(None, None, "STT model not available")
            return
            
        if not self.agent:
            logger.error("AI agent not available")
            yield AdditionalOutputs(None, None, "AI agent not available")
            return
        
        # Process audio
        try:
            # Convert audio to appropriate format
            input_audio_int16 = audio_to_int16(audio_array)
            output_audio_int16 = audio_to_int16(audio_array.copy())
            
            input_file = audio_to_file(audio_input)
            output_file = audio_to_file((sample_rate, output_audio_int16))
            
            # Transcribe audio
            transcription_result = self.stt_model.transcribe(audio_input)
            new_transcription = transcription_result.get('text', '').strip()

            # new_transcription = self.stt_model.stt(audio_input)
            
            if new_transcription:
                # Update combined transcript
                combined_transcription = f"{transcript}{new_transcription}\n" if transcript else f"{new_transcription}\n"
                logger.info(f"Transcription: '{new_transcription}'")
                
                # Generate AI response
                workflow_input = {"messages": [HumanMessage(content=new_transcription)]}
                config = {"configurable": {"thread_id": thread_id}}
                
                response = self.agent.invoke(workflow_input, config=config)['messages'][-1].content
                logger.info(f"AI Response: '{response}'")
                
                # Generate TTS audio chunks
                if self.tts_model:
                    for audio_chunk in self.tts_model.stream_text_to_speech(response):
                        yield audio_chunk
                else:
                    logger.warning("TTS model not available")
                
                # Yield additional outputs
                yield AdditionalOutputs(input_file, output_file, combined_transcription)
            else:
                logger.info("No transcription detected")
                yield AdditionalOutputs(input_file, output_file, transcript)
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            yield AdditionalOutputs(None, None, f"Error: {str(e)}")

# Global audio processor instance
audio_processor = AudioProcessor()

async def get_credentials():
    """Get Cloudflare TURN credentials for WebRTC"""
    try:
        return await get_cloudflare_turn_credentials_async(hf_token=TOKEN)
    except Exception as e:
        logger.error(f"Failed to get TURN credentials: {e}")
        return None

def handle_new_conversation() -> Tuple[str, str]:
    """
    Handle new conversation button click
    
    Returns:
        Tuple of (new_thread_id, cleared_transcript)
    """
    new_thread_id = audio_processor.start_new_conversation()
    return new_thread_id, ""  # Clear transcript

def create_gradio_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(
        title="Real-time Audio Processing",
        theme=gr.themes.Ocean(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <h1 style='text-align: center; color: #2563eb;'>
            🎤 Real-time Audio Processing with AI Conversation
        </h1>
        """)
        
        with gr.Row():
            # Left column - Audio streaming and controls
            with gr.Column(scale=1):
                gr.Markdown("### 🎙️ Live Audio Stream")
                
                # New conversation button
                new_conversation_btn = gr.Button(
                    "🔄 Start New Conversation",
                    variant="secondary",
                    size="sm"
                )
                
                # Thread ID display
                thread_id_display = gr.Textbox(
                    label="Conversation ID",
                    value=audio_processor.get_current_thread_id(),
                    interactive=False,
                    max_lines=1
                )
                
                # Audio streaming component
                audio_stream = WebRTC(
                    label="Record & Process Audio",
                    mode="send-receive",
                    modality="audio",
                    track_constraints=AUDIO_CONSTRAINTS,
                    rtc_configuration=get_credentials,
                    min_width=80,
                )
            
            # Right column - Results and audio files
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Conversation Transcript")
                transcript = gr.Textbox(
                    label="Transcription",
                    placeholder="Your conversation will appear here...",
                    lines=8,
                    max_lines=15,
                    show_copy_button=True
                )
                
                gr.Markdown("### 🔊 Audio Files")
                with gr.Row():
                    input_player = gr.Audio(
                        label="Input Audio", 
                        interactive=False,
                        show_download_button=True
                    )
                    output_player = gr.Audio(
                        label="Processed Audio", 
                        interactive=False,
                        show_download_button=True
                    )
        
       
        # Event handlers
        new_conversation_btn.click(
            fn=handle_new_conversation,
            outputs=[thread_id_display, transcript],
            show_progress="hidden"
        )
        
        # Set up audio streaming with VAD
        hum_vad_model = HumAwareVADModel()
        audio_stream.stream(
            fn=ReplyOnPause(
                fn=audio_processor.process_audio_input,
                input_sample_rate=16000,
                algo_options=ALGO_OPTIONS,
                model_options=VAD_OPTIONS,
                # model=hum_vad_model,
                can_interrupt=True,
            ),
            inputs=[audio_stream, transcript],
            outputs=[audio_stream],
            time_limit=300,
            concurrency_limit=5,
        )
        
        # Handle additional outputs (audio files and transcript updates)
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

def main():
    """Main function to run the application"""
    print("🚀 Starting Real-time Audio Processing Application...")
    
    # Check model initialization status
    models_status = {
        "STT": audio_processor.stt_model is not None,
        "TTS": audio_processor.tts_model is not None,
        "AI Agent": audio_processor.agent is not None
    }
    
    print("\n📊 Model Initialization Status:")
    for model_name, status in models_status.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {model_name}: {'Ready' if status else 'Failed'}")
    
    if not any(models_status.values()):
        print("\n⚠️  WARNING: No models initialized successfully. Check logs for details.")
        return
    
    print(f"\n🎯 Initial conversation ID: {audio_processor.get_current_thread_id()}")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch(
        share=False,
        server_port=7862,
        server_name="0.0.0.0",
        show_error=True
    )

if __name__ == "__main__":
    main()