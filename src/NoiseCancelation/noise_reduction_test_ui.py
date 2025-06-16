"""
Simplified Gradio UI for testing FastRTC noise cancellation models

Tests: All ClearVoice models, DeepFilterNet3, and Spectral Subtraction

Usage:
    python noise_reduction_test_ui.py

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

# Import our FastRTC noise reduction module
from fastrtc_noise_reduction import FastRTCNoiseReduction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastRTCNoiseTester:
    """Simplified test interface for FastRTC noise reduction models"""
    
    def __init__(self):
        self.models = {}
        self.model_status = {}
        
    def load_model(self, model_type: str) -> str:
        """Load a specific noise reduction model"""
        try:
            if model_type in self.models and self.models[model_type] is not None:
                return f"✅ {model_type} already loaded"
            
            print(f"Loading {model_type}...")
            # Only try the requested model, no fallback
            noise_reducer = FastRTCNoiseReduction(model_type)
            
            if noise_reducer.model and noise_reducer.model.is_ready:
                self.models[model_type] = noise_reducer
                info = noise_reducer.get_model_info()
                return f"✅ {model_type} loaded successfully! ({info['model_name']} @ {info['sample_rate']}Hz)"
            else:
                return f"❌ {model_type} failed to initialize"
                
        except Exception as e:
            logger.error(f"Error loading {model_type}: {e}")
            return f"❌ Error loading {model_type}: {str(e)}"
    
    def process_audio(self, audio_file, model_type: str) -> Tuple[Optional[Tuple], str]:
        """Process audio with selected model"""
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
            
            # Process audio
            start_time = time.time()
            enhanced_audio = self.models[model_type].process_audio_chunk(audio, sr)
            processing_time = time.time() - start_time
            
            # Get model info
            info = self.models[model_type].get_model_info()
            
            # Calculate performance metrics
            rtf = processing_time / (original_length / sr)  # Real-time factor
            
            status_msg = (
                f"✅ Processed in {processing_time:.2f}s | "
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
tester = FastRTCNoiseTester()

# Model configurations - Core FastRTC models only
MODELS = {
    "deepfilternet3": "DeepFilterNet3 - Latest real-time 48kHz (Excellent)",
    "mossformer2_se_48k": "MossFormer2_SE_48K - SOTA Transformer 48kHz (Best Quality)",
    "frcrn_se_16k": "FRCRN_SE_16K - Real-time CNN 16kHz (Fast & Good)",
    "mossformergan_se_16k": "MossFormerGAN_SE_16K - GAN-based 16kHz (High Quality)",
    
}

NOISE_TYPES = {
    "white": "White Noise - Equal power across all frequencies",
    "pink": "Pink Noise - 1/f power spectrum (more natural)",
    "babble": "Babble Noise - Simulated background conversation"
}

def create_interface():
    """Create optimized Gradio interface"""
    title="FastRTC Noise Reduction Tester",
    theme = gr.themes.Ocean()
    css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    with gr.Blocks(
        title=title, 
        theme=theme,
        css=css) as demo:
        # Header
        gr.Markdown("# 🎧 FastRTC Noise Reduction Tester")
        gr.Markdown("**Test core ClearVoice models + DeepFilterNet3 for real-time audio processing**")
        
        # Main control row
        with gr.Row():
            # Model selection
            model_selector = gr.Radio(
                choices=list(MODELS.keys()),
                value="deepfilternet3",
                label="🤖 Select Model",
                info="Choose noise reduction model"
            )
            
            # Quick actions
            with gr.Column():
                with gr.Row():
                    load_model_btn = gr.Button("📥 Load Model", variant="secondary", size="sm")
                    process_btn = gr.Button("🎯 Process Audio", variant="primary", size="lg")
                
                # Status
                status_text = gr.Textbox(label="Status", interactive=False, lines=1)
        
        # Audio section
        with gr.Row():
            # Input column
            with gr.Column():
                gr.Markdown("### 📤 Input")
                audio_input = gr.Audio(
                    label="Upload Audio (.wav, .mp3, etc.)",
                    type="filepath",
                    show_download_button=True
                )
                # Examples section
                with gr.Row():
                    gr.Examples(
                        examples=[
                            ["./samples/krisp-original.wav"],
                            ["./samples/vocal_noise_1.wav"]
                        ],
                        inputs=audio_input,
                        cache_examples=False,
                    )
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
                    gr.Markdown("**Performance**")
                    with gr.Row():
                        rtf_display = gr.Textbox(label="RTF", interactive=False, scale=1)
                        device_display = gr.Textbox(label="Device", interactive=False, scale=1)
                        quality_display = gr.Textbox(label="Quality", interactive=False, scale=1)
        
        # Comparison section
        with gr.Row():
            original_audio = gr.Audio(label="📤 Original", scale=1)
            noisy_audio = gr.Audio(label="🔊 With Noise", scale=1)
        
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
            4. **Process** → Compare results
            
            ### 📊 Performance Guide:
            - **RTF < 1.0**: Real-time capable ✅
            - **RTF > 1.0**: Too slow for real-time ❌
            - **Quality**: MossFormer2 > MossFormerGAN > FRCRN > DeepFilterNet3
            - **Speed**: DeepFilterNet3 > FRCRN > MossFormerGAN > MossFormer2
            """)
        
        # Event handlers with enhanced feedback
        def show_original(audio_file):
            if audio_file:
                try:
                    audio, sr = librosa.load(audio_file, sr=None)
                    return (sr, audio), "📤 Original audio loaded"
                except Exception as e:
                    return None, f"❌ Error loading audio: {str(e)}"
            return None, "No audio file"
        
        def preload_model(model_type):
            return tester.load_model(model_type)
        
        def process_with_metrics(audio_file, model_type):
            result, status = tester.process_audio(audio_file, model_type)
            
            # Parse status for metrics display
            rtf = "N/A"
            device = "N/A" 
            quality = "N/A"
            
            if "RTF:" in status:
                try:
                    rtf_part = status.split("RTF: ")[1].split("x")[0]
                    rtf = f"{float(rtf_part):.2f}x"
                    rtf_color = "🟢" if float(rtf_part) < 1.0 else "🔴"
                    rtf = f"{rtf_color} {rtf}"
                except:
                    pass
            
            if "Device:" in status:
                try:
                    device = status.split("Device: ")[1].split(" ")[0]
                    device = f"🖥️ {device}"
                except:
                    pass
                    
            # Quality estimation based on model
            quality_map = {
                "mossformer2_se_48k": "🌟 Excellent",
                "mossformergan_se_16k": "⭐ Very Good", 
                "frcrn_se_16k": "✨ Good",
                "deepfilternet3": "💫 Excellent"
            }
            quality = quality_map.get(model_type, "N/A")
            
            return result, status, rtf, device, quality
        
        # Auto-show original when uploaded
        audio_input.change(
            fn=show_original,
            inputs=[audio_input],
            outputs=[original_audio, status_text]
        )
        
        # Pre-load model button
        load_model_btn.click(
            fn=preload_model,
            inputs=[model_selector],
            outputs=[status_text]
        )
        
        # Add noise button
        add_noise_btn.click(
            fn=tester.add_noise,
            inputs=[audio_input, noise_type, noise_level],
            outputs=[noisy_audio, status_text]
        )
        
        # Process audio button with metrics
        process_btn.click(
            fn=process_with_metrics,
            inputs=[audio_input, model_selector],
            outputs=[enhanced_audio, status_text, rtf_display, device_display, quality_display]
        )
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting FastRTC Noise Reduction Tester...")
    print("📋 Available models:")
    for key, desc in MODELS.items():
        print(f"  - {key}: {desc}")
    
    print("\n🔊 Available noise types:")
    for key, desc in NOISE_TYPES.items():
        print(f"  - {key}: {desc}")
    
    demo = create_interface()
    
    # Launch interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=False,
        show_error=True,
        debug=True
    )