"""
Optimized Gradio UI for testing FastRTC noise cancellation models
Uses direct tensor processing for maximum performance

Tests: All ClearVoice models, DeepFilterNet3
Usage:
    python optimized_noise_reduction_test_ui.py

Author: AI Assistant
Date: 2025
"""

import gradio as gr
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
import time
import logging

# Import our optimized FastRTC noise reduction module
from src.NoiseCancelation.fastrtc_noise_reduction import FastRTCNoiseReduction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedFastRTCNoiseTester:
    """Optimized test interface for FastRTC noise reduction models using direct tensor processing"""
    
    def __init__(self):
        self.models = {}
        self.model_status = {}
        
    def load_model(self, model_type: str) -> str:
        """Load a specific noise reduction model"""
        try:
            if model_type in self.models and self.models[model_type] is not None:
                return f"✅ {model_type} already loaded"
            
            print(f"Loading {model_type}...")
            # Initialize with direct tensor processing
            noise_reducer = FastRTCNoiseReduction(model_type)
            
            if noise_reducer.model and noise_reducer.model.is_ready:
                self.models[model_type] = noise_reducer
                info = noise_reducer.get_model_info()
                return f"✅ {model_type} loaded successfully! ({info['model_name']} @ {info['sample_rate']}Hz on {info['device']})"
            else:
                return f"❌ {model_type} failed to initialize"
                
        except Exception as e:
            logger.error(f"Error loading {model_type}: {e}")
            return f"❌ Error loading {model_type}: {str(e)}"
    
    def process_audio(self, audio_file, model_type: str) -> Tuple[Optional[Tuple], str]:
        """Process audio with selected model using direct tensor processing"""
        if audio_file is None:
            return None, "Please upload an audio file"
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=None)
            original_length = len(audio)
            
            # Ensure model is loaded
            if model_type not in self.models or self.models[model_type] is None:
                status = self.load_model(model_type)
                if "❌" in status:
                    return None, status
            
            # Process audio using direct tensor processing (no file I/O)

            start_time = time.time()
            enhanced_audio = self.models[model_type].process_audio_chunk(audio, sr)
            processing_time = time.time() - start_time
            
            # Get model info
            info = self.models[model_type].get_model_info()
            
            # Calculate performance metrics
            rtf = processing_time / (original_length / sr)  # Real-time factor
            
            
            status_msg = (
                f"✅ Processed in {processing_time:.3f}s | "
                f"RTF: {rtf:.2f}x | "
                f"Model: {info['model_name']} | "
                f"SR: {info['sample_rate']}Hz | "
                f"Device: {info['device']}"
            )
            
            return (sr, enhanced_audio), status_msg
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return None, f"❌ Error: {str(e)}"
    
    def add_noise(self, audio_file, noise_type: str, noise_level: float) -> Tuple[Optional[Tuple], str]:
        """Add different types of noise for testing"""
        if audio_file is None:
            return None, "Please upload an audio file"
        
        try:
            audio, sr = librosa.load(audio_file, sr=None)
            
            if noise_type == "white":
                noise = np.random.normal(0, noise_level, len(audio))
            elif noise_type == "pink":
                # Simple pink noise approximation
                white_noise = np.random.normal(0, noise_level, len(audio))
                # Apply a simple filter to approximate pink noise
                noise = np.convolve(white_noise, [1, -0.5], mode='same')
            elif noise_type == "babble":
                # Simulate babble noise with multiple sine waves
                t = np.linspace(0, len(audio)/sr, len(audio))
                noise = np.zeros_like(audio)
                for freq in [200, 300, 500, 800, 1200]:
                    noise += noise_level * 0.2 * np.sin(2 * np.pi * freq * t)
                noise += np.random.normal(0, noise_level * 0.3, len(audio))
            else:
                noise = np.random.normal(0, noise_level, len(audio))
            
            noisy_audio = audio + noise
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(noisy_audio))
            if max_val > 1.0:
                noisy_audio = noisy_audio / max_val
            
            # Calculate SNR
            signal_power = np.mean(audio ** 2)
            noise_power = np.mean(noise ** 2)
            snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            
            return (sr, noisy_audio), f"✅ Added {noise_type} noise (level: {noise_level:.2f}, SNR: {snr_db:.1f} dB)"
            
        except Exception as e:
            logger.error(f"Noise addition error: {e}")
            return None, f"❌ Error: {str(e)}"

# Create tester
tester = OptimizedFastRTCNoiseTester()

# Model configurations - Core FastRTC models
MODELS = {
    "deepfilternet3": "DeepFilterNet3 - Real-time 48kHz (Excellent & Fast)",
    "mossformer2_se_48k": "MossFormer2_SE_48K - SOTA Transformer 48kHz (Best Quality)",
    "frcrn_se_16k": "FRCRN_SE_16K - Real-time CNN 16kHz (Fast & Good)",
    "mossformergan_se_16k": "MossFormerGAN_SE_16K - GAN-based 16kHz (High Quality)",
    # "mossformer2_ss_16k": "MossFormer2_SS_16K - Speech Separation 16kHz (Multi-speaker)",
}

NOISE_TYPES = {
    "white": "White Noise - Equal power across all frequencies",
    "pink": "Pink Noise - 1/f power spectrum (more natural)",
    "babble": "Babble Noise - Simulated background conversation"
}

def create_interface():
    """Create optimized Gradio interface"""
    title = "FastRTC Noise Reduction Tester (Optimized)"
    theme = gr.themes.Ocean()
    css = """
        footer {
            visibility: hidden;
        }
        .performance-metric {
            background: linear-gradient(45deg, #1e3a8a, #3b82f6);
            color: white;
            padding: 8px;
            border-radius: 6px;
            font-weight: bold;
        }
        """
    
    with gr.Blocks(
        title=title, 
        theme=theme,
        css=css,
        fill_width=True) as demo:
        # Header
        gr.Markdown("# 🚀 Noise Reduction Tester")
        
        # Model info section
        with gr.Accordion("📋 Model Information", open=False):
            with gr.Row():
                for key, desc in MODELS.items():
                    gr.Markdown(f"**{key}**\n{desc}")
        
        # Quick tips
        with gr.Accordion("💡 Quick Tips", open=False):
            gr.Markdown("""
            ### 🚀 Testing Workflow:
            1. **Upload speech audio** → See original
            2. **Add noise** (optional) → Test robustness  
            3. **Select model** → Choose quality vs speed
            4. **Process** → Compare results & performance
            
            ### 📊 Performance Metrics:
            - **RTF < 0.3**: Excellent real-time performance ⚡
            - **RTF < 1.0**: Real-time capable ✅  
            - **RTF > 1.0**: Too slow for real-time ❌
                        
            ### 🏆 Quality vs Speed:
            - **Best Quality**: MossFormer2_SE_48K > MossFormerGAN > FRCRN
            - **Best Speed**:  DeepFilterNet3 > FRCRN > MossFormerGAN > MossFormer2
            - **Best Balance**: DeepFilterNet3 (recommended for most use cases)
            """)
        
        # Main control row
        with gr.Row():
            # Model selection
            model_selector = gr.Radio(
                choices=list(MODELS.keys()),
                value="deepfilternet3",
                label="🤖 Select Noise Reduction Model",
            )
            
            # Quick actions
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        load_model_btn = gr.Button("📥 Load Model", variant="secondary", size="sm")
                        process_btn = gr.Button("⚡ Process Audio", variant="primary", size="lg")
                
                    # Status
                    status_text = gr.Textbox(label="Status", interactive=False, lines=1)
        
        # Audio section
        with gr.Row():
            # Input column
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📤 Input")
                        audio_input = gr.Audio(
                            label="📤 Original",
                            type="filepath",
                            show_download_button=True,
                            scale=1
                        )
                        # Examples section
                        with gr.Row():
                            gr.Examples(
                                examples=[
                                    ["./src/NoiseCancelation/samples/krisp-original.wav"],
                                    ["./src/NoiseCancelation/samples/vocal_noise_1.wav"]
                                ],
                                label="Sample Audio Files",
                                inputs=audio_input,
                                cache_examples=False,
                            )
                    with gr.Column():
                        gr.Markdown("### 🔊 Add Noise")
                        noisy_audio = gr.Audio(label="🔊 With Noise", scale=1)
                
                        # Noise controls
                        with gr.Group():
                            gr.Markdown("**Add Test Noise**")
                            with gr.Row():
                                noise_type = gr.Dropdown(
                                    choices=list(NOISE_TYPES.keys()),
                                    value="white",
                                    label="Type",
                                    scale=1
                                )
                                noise_level = gr.Slider(
                                    minimum=0.01,
                                    maximum=0.3,
                                    value=0.1,
                                    step=0.01,
                                    label="Level",
                                    scale=2
                                )
                            add_noise_btn = gr.Button("➕ Add Noise", variant="secondary", size="sm")
            
            # Output column
            with gr.Column():
                gr.Markdown("### ✨ Output")
                enhanced_audio = gr.Audio(label="Enhanced Audio")
                
                # Performance metrics
                with gr.Group():
                    gr.Markdown("**Performance Metrics**")
                    with gr.Row():
                        rtf_display = gr.Textbox(label="RTF", interactive=False, scale=1)
                        device_display = gr.Textbox(label="Device", interactive=False, scale=1)
                        time_display = gr.Textbox(label="Time", interactive=False, scale=1)
        
       
        
        # Event handlers with enhanced feedback
        def original_change(audio_file):
            if audio_file:
                return None, None, "📤 Original audio loaded - ready for processing"
            return None, None, "No audio file"
        
        def preload_model(model_type):
            return tester.load_model(model_type)
        
        def process_with_metrics(audio_file, model_type):
            result, status = tester.process_audio(audio_file, model_type)
            
            # Parse status for metrics display
            rtf = "N/A"
            device = "N/A" 
            time_taken = "N/A"
            
            if "RTF:" in status:
                try:
                    rtf_part = status.split("RTF: ")[1].split("x")[0]
                    rtf_val = float(rtf_part)
                    rtf_color = "🟢" if rtf_val < 0.3 else "🟡" if rtf_val < 1.0 else "🔴"
                    rtf = f"{rtf_color} {rtf_val:.3f}x"
                except:
                    pass
            
            
            if "Device:" in status:
                try:
                    device_part = status.split("Device: ")[1].split(" ")[0]
                    device = f"🖥️ {device_part}"
                except:
                    pass
            
            if "Processed in" in status:
                try:
                    time_part = status.split("Processed in ")[1].split("s")[0]
                    time_taken = f"⏱️ {float(time_part):.3f}s"
                except:
                    pass
            
            return result, status, rtf, device, time_taken
        
        def add_noise_with_feedback(audio_file, noise_type, noise_level):
            result, status = tester.add_noise(audio_file, noise_type, noise_level)
            return result, status, None
        
        # Auto-show original when uploaded
        audio_input.change(
            fn=original_change,
            inputs=[audio_input],
            outputs=[noisy_audio, enhanced_audio, status_text]
        )
        
        # Pre-load model button
        load_model_btn.click(
            fn=preload_model,
            inputs=[model_selector],
            outputs=[status_text]
        )
        
        # Add noise button
        add_noise_btn.click(
            fn=add_noise_with_feedback,
            inputs=[audio_input, noise_type, noise_level],
            outputs=[noisy_audio, status_text, enhanced_audio]
        )
        
        # Process audio button with comprehensive metrics
        process_btn.click(
            fn=process_with_metrics,
            inputs=[audio_input, model_selector],
            outputs=[enhanced_audio, status_text, rtf_display, device_display, time_display]
        )
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting Optimized FastRTC Noise Reduction Tester...")
    print("⚡ Using direct tensor processing for maximum performance!")
    print()
    print("📋 Available models:")
    for key, desc in MODELS.items():
        print(f"  - {key}: {desc}")
    
    print("\n🔊 Available noise types:")
    for key, desc in NOISE_TYPES.items():
        print(f"  - {key}: {desc}")
    
    print("\n🎯 Performance Features:")
    print("  - Direct tensor processing (no file I/O)")
    print("  - Automatic GPU detection and usage")
    print("  - Real-time performance monitoring")
    print("  - Memory-efficient processing")
    
    demo = create_interface()
    
    # Launch interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=True,
        show_error=True,
        debug=True
    )