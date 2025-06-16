"""
FastRTC Integration Test UI for Noise Cancellation
Tests real-time audio streaming with noise reduction using actual FastRTC components

Usage:
    python fastrtc_integration_test_ui.py

Author: AI Assistant
Date: 2025
"""

import gradio as gr
import numpy as np
import asyncio
import time
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

# FastRTC imports
from fastrtc import Stream, StreamHandler, AsyncStreamHandler
from fastrtc_noise_reduction import FastRTCNoiseReduction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NoiseReductionStats:
    """Performance statistics for noise reduction"""
    frames_processed: int = 0
    total_processing_time: float = 0.0
    total_audio_duration: float = 0.0
    avg_rtf: float = 0.0
    last_rtf: float = 0.0
    errors: int = 0

class NoiseReductionStreamHandler(StreamHandler):
    """Synchronous FastRTC StreamHandler with noise reduction"""
    
    def __init__(self, model_type: str = "mossformer2_se_48k"):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=48000,
            input_sample_rate=48000
        )
        self.model_type = model_type
        self.noise_reducer: Optional[FastRTCNoiseReduction] = None
        self.stats = NoiseReductionStats()
        
    def start_up(self):
        """Initialize noise reduction model on startup"""
        try:
            logger.info(f"Initializing {self.model_type} model...")
            self.noise_reducer = FastRTCNoiseReduction(self.model_type)
            logger.info(f"✅ Model {self.model_type} ready!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.model_type}: {e}")
            self.noise_reducer = None
    
    def receive(self, frame: Tuple[int, np.ndarray]) -> None:
        """Process incoming audio frame"""
        sample_rate, audio_data = frame
        
        if self.noise_reducer is None:
            return
        
        try:
            # Convert int16 to float32 for processing
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Apply noise reduction
            start_time = time.time()
            enhanced_audio = self.noise_reducer.process_audio_chunk(audio_float, sample_rate)
            processing_time = time.time() - start_time
            
            # Update stats
            audio_duration = len(audio_data) / sample_rate
            rtf = processing_time / audio_duration if audio_duration > 0 else 0
            
            self.stats.frames_processed += 1
            self.stats.total_processing_time += processing_time
            self.stats.total_audio_duration += audio_duration
            self.stats.last_rtf = rtf
            self.stats.avg_rtf = (self.stats.total_processing_time / 
                                self.stats.total_audio_duration if self.stats.total_audio_duration > 0 else 0)
            
            # Convert back to int16 and store for emit
            enhanced_int16 = (enhanced_audio * 32768.0).astype(np.int16)
            self.processed_frame = (sample_rate, enhanced_int16)
            
        except Exception as e:
            logger.error(f"Error in noise reduction: {e}")
            self.stats.errors += 1
            # Fallback to original audio
            self.processed_frame = frame
    
    def emit(self):
        """Return processed audio frame"""
        if hasattr(self, 'processed_frame'):
            return self.processed_frame
        return None
    
    def copy(self):
        """Create a copy for new connections"""
        return NoiseReductionStreamHandler(self.model_type)

class AsyncNoiseReductionStreamHandler(AsyncStreamHandler):
    """Asynchronous FastRTC StreamHandler with noise reduction"""
    
    def __init__(self, model_type: str = "mossformer2_se_48k"):
        super().__init__(
            expected_layout="mono", 
            output_sample_rate=48000,
            input_sample_rate=48000
        )
        self.model_type = model_type
        self.noise_reducer: Optional[FastRTCNoiseReduction] = None
        self.stats = NoiseReductionStats()
        self.audio_queue = asyncio.Queue()
        
    async def start_up(self):
        """Initialize noise reduction model on startup"""
        try:
            logger.info(f"Initializing {self.model_type} model...")
            self.noise_reducer = FastRTCNoiseReduction(self.model_type)
            logger.info(f"✅ Model {self.model_type} ready!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.model_type}: {e}")
            self.noise_reducer = None
    
    async def receive(self, frame: Tuple[int, np.ndarray]) -> None:
        """Process incoming audio frame asynchronously"""
        sample_rate, audio_data = frame
        
        if self.noise_reducer is None:
            await self.audio_queue.put(frame)
            return
        
        try:
            # Convert int16 to float32 for processing
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Apply noise reduction
            start_time = time.time()
            enhanced_audio = self.noise_reducer.process_audio_chunk(audio_float, sample_rate)
            processing_time = time.time() - start_time
            
            # Update stats
            audio_duration = len(audio_data) / sample_rate
            rtf = processing_time / audio_duration if audio_duration > 0 else 0
            
            self.stats.frames_processed += 1
            self.stats.total_processing_time += processing_time
            self.stats.total_audio_duration += audio_duration
            self.stats.last_rtf = rtf
            self.stats.avg_rtf = (self.stats.total_processing_time / 
                                self.stats.total_audio_duration if self.stats.total_audio_duration > 0 else 0)
            
            # Convert back to int16
            enhanced_int16 = (enhanced_audio * 32768.0).astype(np.int16)
            processed_frame = (sample_rate, enhanced_int16)
            
            await self.audio_queue.put(processed_frame)
            
        except Exception as e:
            logger.error(f"Error in noise reduction: {e}")
            self.stats.errors += 1
            # Fallback to original audio
            await self.audio_queue.put(frame)
    
    async def emit(self):
        """Return processed audio frame"""
        try:
            return await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    def copy(self):
        """Create a copy for new connections"""
        return AsyncNoiseReductionStreamHandler(self.model_type)

class FastRTCIntegrationTester:
    """Main integration tester using real FastRTC components"""
    
    def __init__(self):
        self.current_stream: Optional[Stream] = None
        self.handler_stats: Optional[NoiseReductionStats] = None
        
    def create_stream(self, model_type: str, handler_type: str) -> Tuple[Stream, str]:
        """Create FastRTC stream with noise reduction"""
        try:
            # Stop existing stream
            if self.current_stream:
                # Note: In real usage, you'd properly cleanup the stream
                pass
            
            # Create handler
            if handler_type == "sync":
                handler = NoiseReductionStreamHandler(model_type)
            else:  # async
                handler = AsyncNoiseReductionStreamHandler(model_type)
            
            # Create FastRTC stream
            stream = Stream(
                handler=handler,
                modality="audio",
                mode="send-receive",
                additional_inputs=[
                    gr.Textbox(label="Model Type", value=model_type, interactive=False),
                    gr.Textbox(label="Handler Type", value=handler_type, interactive=False)
                ]
            )
            
            self.current_stream = stream
            self.handler_stats = handler.stats
            
            return stream, f"✅ Created {handler_type} FastRTC stream with {model_type}"
            
        except Exception as e:
            logger.error(f"Error creating stream: {e}")
            return None, f"❌ Error: {str(e)}"
    
    def get_stats(self) -> str:
        """Get current performance statistics"""
        if not self.handler_stats:
            return "No active stream"
        
        stats = self.handler_stats
        return f"""
📊 Performance Statistics:
• Frames processed: {stats.frames_processed}
• Total audio: {stats.total_audio_duration:.1f}s
• Average RTF: {stats.avg_rtf:.3f}x
• Last RTF: {stats.last_rtf:.3f}x
• Errors: {stats.errors}
• Real-time capable: {'✅' if stats.avg_rtf < 1.0 else '❌'}
        """.strip()

def create_integration_test_ui():
    """Create the FastRTC integration test UI"""
    
    tester = FastRTCIntegrationTester()
    
    # Available models and handler types
    MODELS = [
        "mossformer2_se_48k",
        "frcrn_se_16k", 
        "mossformergan_se_16k",
        "deepfilternet3"
    ]
    
    HANDLER_TYPES = ["sync", "async"]
    
    with gr.Blocks(title="FastRTC Noise Cancellation Integration Test", theme=gr.themes.Soft()) as demo:
        # Header
        gr.Markdown("# 🎧 FastRTC Noise Cancellation Integration Test")
        gr.Markdown("**Test real-time noise reduction with actual FastRTC StreamHandler components**")
        
        # Configuration section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚙️ Configuration")
                model_selector = gr.Dropdown(
                    choices=MODELS,
                    value="mossformer2_se_48k",
                    label="Noise Reduction Model"
                )
                handler_type = gr.Radio(
                    choices=HANDLER_TYPES,
                    value="async",
                    label="Handler Type"
                )
                create_stream_btn = gr.Button("🚀 Create Stream", variant="primary")
                
            with gr.Column():
                gr.Markdown("### 📊 Performance Monitor")
                stats_display = gr.Textbox(
                    label="Performance Statistics",
                    value="No active stream",
                    lines=8,
                    interactive=False
                )
                refresh_stats_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
        
        # Status section
        status_text = gr.Textbox(label="Status", interactive=False)
        
        # Stream section
        with gr.Row():
            gr.Markdown("### 🎙️ FastRTC Stream")
            gr.Markdown("After creating a stream, you can launch it separately or integrate it into your application")
        
        # Instructions for manual testing
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🧪 Manual Testing Steps")
                gr.Markdown("""
                1. Click "Create Stream" above
                2. Copy the handler code for your application
                3. The stream object is ready for `.ui.launch()` or `.mount(app)`
                """)
            
            with gr.Column():
                gr.Markdown("### 💻 Code Example")
                code_example = gr.Code(
                    value="""
# Example integration code:
from fastrtc import Stream

# Create your handler (from the UI above)
handler = NoiseReductionStreamHandler("mossformer2_se_48k")

# Create stream
stream = Stream(
    handler=handler,
    modality="audio", 
    mode="send-receive"
)

# Option 1: Launch UI
stream.ui.launch()

# Option 2: Mount on FastAPI
from fastapi import FastAPI
app = FastAPI()
stream.mount(app)
""",
                    language="python",
                    label="Integration Code"
                )
        
        # Model information
        with gr.Accordion("📋 Model Information", open=False):
            gr.Markdown("""
            ### Available Models:
            - **mossformer2_se_48k**: SOTA Transformer 48kHz (Best Quality)
            - **frcrn_se_16k**: Real-time CNN 16kHz (Fast & Good)  
            - **mossformergan_se_16k**: GAN-based 16kHz (High Quality)
            - **deepfilternet3**: Latest real-time 48kHz (Excellent Balance)
            
            ### Handler Types:
            - **Sync**: Synchronous StreamHandler (simpler, blocking)
            - **Async**: Asynchronous StreamHandler (better for I/O operations)
            """)
        
        # Instructions
        with gr.Accordion("💡 How to Test", open=False):
            gr.Markdown("""
            ### 🚀 Testing Steps:
            1. **Select model and handler type** from the configuration
            2. **Click "Create Stream"** to initialize FastRTC with noise reduction
            3. **Use your microphone** to test real-time noise cancellation
            4. **Monitor performance** with the stats panel
            5. **Check RTF values** - should be < 1.0 for real-time capability
            
            ### 📊 Key Metrics:
            - **RTF (Real-time Factor)**: Processing time / audio duration
            - **RTF < 1.0**: ✅ Real-time capable
            - **RTF > 1.0**: ❌ Too slow for real-time
            - **Errors**: Failed processing attempts
            """)
        
        # Event handlers
        def create_stream_handler(model_type, handler_type):
            stream, status = tester.create_stream(model_type, handler_type)
            return status
        
        def refresh_stats_handler():
            return tester.get_stats()
        
        # Connect events
        create_stream_btn.click(
            fn=create_stream_handler,
            inputs=[model_selector, handler_type],
            outputs=[status_text]
        )
        
        refresh_stats_btn.click(
            fn=refresh_stats_handler,
            outputs=[stats_display]
        )
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting FastRTC Noise Cancellation Integration Test...")
    print("📋 Available models: mossformer2_se_48k, frcrn_se_16k, mossformergan_se_16k, deepfilternet3")
    print("⚙️ Handler types: sync (StreamHandler), async (AsyncStreamHandler)")
    
    demo = create_integration_test_ui()
    
    # Launch the test UI
    demo.launch(
        server_name="0.0.0.0",
        server_port=7866,
        share=False,
        show_error=True,
        debug=True
    )