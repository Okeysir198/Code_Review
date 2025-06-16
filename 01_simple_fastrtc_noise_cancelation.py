"""
Real-time Audio Processing Application with Enhanced Features
Uses EnhancedReplyOnPause with noise cancellation and turn detection
Optimized UI with audio processing insights and conversation tracking
"""

import os
import logging
import uuid
from typing import Generator, Tuple, Dict, Any
import numpy as np
import gradio as gr
from dotenv import find_dotenv, load_dotenv

# FastRTC imports
from fastrtc import (
    WebRTC, 
    AlgoOptions,
    SileroVadOptions,
    AdditionalOutputs, 
    get_cloudflare_turn_credentials_async,
    audio_to_int16, 
    audio_to_file,
    get_stt_model
)

# Enhanced imports
from src.VAD_TurnDectection.enhanced_reply_on_pause import (
    EnhancedReplyOnPause,
    NoiseReductionConfig,
    TurnDetectionConfig,
    create_enhanced_handler
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

# Enhanced configurations
ENHANCED_CONFIG = {
    "noise_reduction": {
        "enabled": True,
        "model": "deepfilternet3",
        "fallback_enabled": True
    },
    "turn_detection": {
        "enabled": True,
        "model": "livekit",
        "confidence_threshold": 0.5,
        "fallback_to_pause": True
    }
}

# Model configurations (simplified)
STT_CONFIG = {
    "model_name": "nvidia/parakeet-tdt-0.6b-v2",
    "nvidia/parakeet-tdt-0.6b-v2": {
        "timestamp_prediction": True,
        "decoding_type": "tdt"
    }
}

TTS_CONFIG = {
    "model_name": "kokorov2",
    "kokorov2": {
        "voice": "am_michael",
        "speed": 1.3,
        "language": "a",
        "use_gpu": True,
        "sample_rate": 24000,
    }
}

# WebRTC constraints
AUDIO_CONSTRAINTS = {
    "noiseSuppression": {"exact": True},
    "autoGainControl": {"exact": True},
    "sampleRate": {"ideal": 16000},
    "channelCount": {"exact": 1},
}

# Audio processing options
VAD_OPTIONS = SileroVadOptions(
    threshold=0.5,
    min_speech_duration_ms=250,
    max_speech_duration_s=30,
    min_silence_duration_ms=500,
    window_size_samples=1024,
    speech_pad_ms=50,
)

ALGO_OPTIONS = AlgoOptions(
    audio_chunk_duration=3.0,  # Longer for semantic analysis
    started_talking_threshold=0.2,
    speech_threshold=0.1,
)


class EnhancedAudioProcessor:
    """Enhanced audio processor with conversation tracking and processing insights"""
    
    def __init__(self):
        self.stt_model = None
        self.fastrtc_stt_model = None  # For turn detection
        self.tts_model = None
        self.llm_model = None
        self.agent: CompiledGraph = None
        self.checkpointer = MemorySaver()
        self.current_thread_id = str(uuid.uuid4())
        self.conversation_history = []
        self.enhanced_handler = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models"""
        try:
            self._initialize_llm()
            self._initialize_stt_models()
            self._initialize_tts_model()
            logger.info("All models initialized successfully")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
    
    def _initialize_llm(self):
        """Initialize LLM and agent"""
        try:
            logger.info("Initializing LLM...")
            self.llm_model = init_chat_model("ollama:qwen2.5:7b-instruct", temperature=0.1)
            
            self.agent = create_react_agent(
                model=self.llm_model,
                tools=[],
                prompt=(
                    "You are a debt collection specialist from Cartrack Account Department. "
                    "Your name is Trung. Keep responses concise (20 words or less). "
                    "Be professional and helpful."
                ),
                checkpointer=self.checkpointer
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            self.llm_model = None
            self.agent = None
    
    def _initialize_stt_models(self):
        """Initialize STT models"""
        try:
            # Your custom STT model
            logger.info("Initializing custom STT model...")
            self.stt_model = create_stt_model(STT_CONFIG)
            
            # FastRTC STT model for turn detection
            logger.info("Initializing FastRTC STT model...")
            self.fastrtc_stt_model = get_stt_model("moonshine/base")
            
            # Warm up
            warmup_audio = np.zeros((16000,), dtype=np.float32)
            input_file = audio_to_file((16000, warmup_audio))
            self.stt_model.transcribe(input_file)
            logger.info("STT models initialized and warmed up")
        except Exception as e:
            logger.error(f"STT initialization failed: {e}")
            self.stt_model = None
            self.fastrtc_stt_model = None
    
    def _initialize_tts_model(self):
        """Initialize TTS model"""
        try:
            logger.info("Initializing TTS model...")
            self.tts_model = create_tts_model(TTS_CONFIG)
            logger.info("TTS model initialized successfully")
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            self.tts_model = None
    
    def start_new_conversation(self) -> Tuple[str, str, str]:
        """Start new conversation and reset history"""
        self.current_thread_id = str(uuid.uuid4())
        self.conversation_history = []
        if self.enhanced_handler:
            self.enhanced_handler.conversation_history = []
        
        logger.info(f"New conversation started: {self.current_thread_id}")
        return self.current_thread_id, "", self._format_conversation()
    
    def _format_conversation(self) -> str:
        """Format conversation history for display"""
        if not self.conversation_history:
            return "Conversation will appear here..."
        
        formatted = []
        for entry in self.conversation_history:
            role = "🗣️ **User**" if entry["role"] == "user" else "🤖 **AI Assistant**"
            formatted.append(f"{role}: {entry['content']}")
        
        return "\n\n".join(formatted)
    
    def process_audio_input(self, audio_input: Tuple[int, np.ndarray], transcript: str = "") -> Generator:
        """Enhanced audio processing with insights"""
        sample_rate, audio_array = audio_input
        logger.info(f"Processing audio: {sample_rate}Hz, {audio_array.shape}")
        
        if not self.stt_model or not self.agent:
            yield AdditionalOutputs(None, None, "Models not available")
            return
        
        try:
            # Convert audio
            input_audio_int16 = audio_to_int16(audio_array)
            
            # Create audio files for different processing stages
            original_file = audio_to_file((sample_rate, input_audio_int16))
            
            # Get noise-reduced audio (simulated - would come from enhanced handler)
            noise_reduced_file = original_file  # Placeholder
            
            # Get VAD-processed audio (simulated)
            vad_processed_file = original_file  # Placeholder
            
            # Transcribe
            transcription_result = self.stt_model.transcribe(original_file)
            new_transcription = transcription_result.get('text', '').strip()
            
            if new_transcription:
                # Add to conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": new_transcription,
                    "timestamp": str(uuid.uuid4())[:8]
                })
                
                # Generate AI response
                workflow_input = {"messages": [HumanMessage(content=new_transcription)]}
                config = {"configurable": {"thread_id": self.current_thread_id}}
                
                response = self.agent.invoke(workflow_input, config=config)['messages'][-1].content
                
                # Add AI response to history
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": response,
                    "timestamp": str(uuid.uuid4())[:8]
                })
                
                # Update enhanced handler conversation history
                if self.enhanced_handler:
                    self.enhanced_handler.add_assistant_response(response)
                
                logger.info(f"User: '{new_transcription}' | AI: '{response}'")
                
                # Generate TTS
                if self.tts_model:
                    for audio_chunk in self.tts_model.stream_text_to_speech(response):
                        yield audio_chunk
                
                # Yield processing insights
                yield AdditionalOutputs(
                    original_file,      # Original audio
                    noise_reduced_file, # Noise-reduced audio
                    vad_processed_file, # VAD-processed audio
                    self._format_conversation()  # Formatted conversation
                )
            else:
                logger.info("No transcription detected")
                yield AdditionalOutputs(original_file, None, None, transcript)
                
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            yield AdditionalOutputs(None, None, None, f"Error: {str(e)}")
    
    def create_enhanced_handler(self) -> EnhancedReplyOnPause:
        """Create enhanced handler with all features"""
        noise_config = NoiseReductionConfig(
            enabled=ENHANCED_CONFIG["noise_reduction"]["enabled"],
            model=ENHANCED_CONFIG["noise_reduction"]["model"],
            fallback_enabled=ENHANCED_CONFIG["noise_reduction"]["fallback_enabled"]
        )
        
        turn_config = TurnDetectionConfig(
            enabled=ENHANCED_CONFIG["turn_detection"]["enabled"],
            model=ENHANCED_CONFIG["turn_detection"]["model"],
            confidence_threshold=ENHANCED_CONFIG["turn_detection"]["confidence_threshold"],
            fallback_to_pause=ENHANCED_CONFIG["turn_detection"]["fallback_to_pause"]
        )
        
        hum_vad_model = HumAwareVADModel()
        
        self.enhanced_handler = EnhancedReplyOnPause(
            fn=self.process_audio_input,
            algo_options=ALGO_OPTIONS,
            model_options=VAD_OPTIONS,
            model=hum_vad_model,
            stt_model=self.fastrtc_stt_model,
            noise_reduction_config=noise_config,
            turn_detection_config=turn_config,
            input_sample_rate=16000,
            can_interrupt=False,
        )
        
        return self.enhanced_handler
    
    def get_system_status(self) -> str:
        """Get system status"""
        status_items = [
            f"🎙️ **STT Model**: {'✅ Ready' if self.stt_model else '❌ Failed'}",
            f"🔊 **TTS Model**: {'✅ Ready' if self.tts_model else '❌ Failed'}",
            f"🤖 **AI Agent**: {'✅ Ready' if self.agent else '❌ Failed'}",
            f"🎯 **FastRTC STT**: {'✅ Ready' if self.fastrtc_stt_model else '❌ Failed'}",
        ]
        
        if self.enhanced_handler:
            handler_status = self.enhanced_handler.get_status()
            noise_status = handler_status.get("noise_reduction", {})
            turn_status = handler_status.get("turn_detection", {})
            
            status_items.extend([
                f"🔇 **Noise Reduction**: {'✅ ' + noise_status.get('model', 'Ready') if noise_status.get('enabled') else '❌ Disabled'}",
                f"🔄 **Turn Detection**: {'✅ ' + turn_status.get('model', 'Ready') if turn_status.get('enabled') else '❌ Disabled'}",
                f"💬 **Conversation**: {len(self.conversation_history)} messages"
            ])
        
        return "\n".join(status_items)


# Global processor instance
audio_processor = EnhancedAudioProcessor()

async def get_credentials():
    """Get TURN credentials"""
    try:
        return await get_cloudflare_turn_credentials_async(hf_token=TOKEN)
    except Exception as e:
        logger.error(f"TURN credentials failed: {e}")
        return {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

def create_gradio_interface():
    """Create optimized Gradio interface"""
    
    with gr.Blocks(
        title="Enhanced FastRTC Audio Processing",
        theme=gr.themes.Ocean(),
        css="""
        .gradio-container { max-width: 1400px !important; }
        .conversation-box { background-color: #f8f9fa; border-radius: 8px; padding: 15px; }
        .status-box { background-color: #e3f2fd; border-radius: 8px; padding: 10px; }
        """
    ) as demo:
        
        gr.HTML("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='color: #1565c0; margin-bottom: 10px;'>
                🎤 Enhanced FastRTC Audio Processing
            </h1>
            <p style='color: #666; font-size: 16px;'>
                Real-time conversation with noise cancellation and turn detection
            </p>
        </div>
        """)
        
        with gr.Row():
            # Left column - Controls and Status
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ **Controls**")
                
                new_conversation_btn = gr.Button(
                    "🔄 Start New Conversation",
                    variant="primary",
                    size="lg"
                )
                
                thread_id_display = gr.Textbox(
                    label="🔗 Conversation ID",
                    value=audio_processor.current_thread_id,
                    interactive=False
                )
                
                gr.Markdown("### 📊 **System Status**")
                system_status = gr.Textbox(
                    label="Status",
                    value=audio_processor.get_system_status(),
                    interactive=False,
                    lines=7,
                    elem_classes=["status-box"]
                )
                
                refresh_status_btn = gr.Button(
                    "🔄 Refresh Status",
                    variant="secondary"
                )
                
                gr.Markdown("### 🎙️ **Audio Stream**")
                audio_stream = WebRTC(
                    label="Voice Chat",
                    mode="send-receive",
                    modality="audio",
                    track_constraints=AUDIO_CONSTRAINTS,
                    rtc_configuration=get_credentials,
                )
            
            # Right column - Conversation and Audio Processing
            with gr.Column(scale=2):
                gr.Markdown("### 💬 **Conversation**")
                conversation_display = gr.Textbox(
                    label="Chat History",
                    value=audio_processor._format_conversation(),
                    lines=12,
                    max_lines=20,
                    show_copy_button=True,
                    elem_classes=["conversation-box"]
                )
                
                gr.Markdown("### 🔊 **Audio Processing Stages**")
                with gr.Row():
                    original_audio = gr.Audio(
                        label="🎵 Original Audio",
                        interactive=False,
                        show_download_button=True
                    )
                    noise_reduced_audio = gr.Audio(
                        label="🔇 Noise Reduced",
                        interactive=False,
                        show_download_button=True
                    )
                    vad_processed_audio = gr.Audio(
                        label="🎯 VAD Processed",
                        interactive=False,
                        show_download_button=True
                    )
        
        # Event handlers
        def handle_new_conversation():
            thread_id, _, conversation = audio_processor.start_new_conversation()
            status = audio_processor.get_system_status()
            return thread_id, conversation, status
        
        new_conversation_btn.click(
            fn=handle_new_conversation,
            outputs=[thread_id_display, conversation_display, system_status],
            show_progress="hidden"
        )
        
        refresh_status_btn.click(
            fn=audio_processor.get_system_status,
            outputs=system_status,
            show_progress="hidden"
        )
        
        # Setup enhanced audio streaming
        enhanced_handler = audio_processor.create_enhanced_handler()
        
        audio_stream.stream(
            fn=enhanced_handler,
            inputs=[audio_stream, conversation_display],
            outputs=[audio_stream],
            time_limit=300,
            concurrency_limit=3,
        )
        
        # Handle processing insights
        audio_stream.on_additional_outputs(
            fn=lambda orig, noise_red, vad_proc, conv: (
                orig,           # Original audio
                noise_red,      # Noise reduced audio  
                vad_proc,       # VAD processed audio
                conv            # Updated conversation
            ),
            outputs=[
                original_audio,
                noise_reduced_audio, 
                vad_processed_audio,
                conversation_display
            ],
            queue=False,
            show_progress="hidden"
        )
    
    return demo

def main():
    """Main application entry point"""
    print("🚀 Starting Enhanced FastRTC Audio Processing...")
    
    # Display initialization status
    status_items = [
        ("STT Model", audio_processor.stt_model is not None),
        ("TTS Model", audio_processor.tts_model is not None), 
        ("AI Agent", audio_processor.agent is not None),
        ("FastRTC STT", audio_processor.fastrtc_stt_model is not None),
    ]
    
    print("\n📊 Initialization Status:")
    for name, status in status_items:
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}: {'Ready' if status else 'Failed'}")
    
    # Enhanced features status
    print(f"\n🔧 Enhanced Features:")
    print(f"  🔇 Noise Reduction: {ENHANCED_CONFIG['noise_reduction']['model']}")
    print(f"  🔄 Turn Detection: {ENHANCED_CONFIG['turn_detection']['model']}")
    print(f"  🎯 Initial Thread: {audio_processor.current_thread_id}")
    
    if not any(status for _, status in status_items):
        print("\n⚠️  WARNING: Critical models failed to initialize!")
        return
    
    # Launch interface
    demo = create_gradio_interface()
    demo.launch(
        share=False,
        server_port=7861,
        server_name="0.0.0.0",
        show_error=True
    )

if __name__ == "__main__":
    main()