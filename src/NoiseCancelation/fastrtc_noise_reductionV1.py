"""
Real-time Noise Cancellation for FastRTC Integration
Optimized for minimum latency and maximum performance.
"""

import numpy as np
import torch
import torchaudio
import tempfile
import os
import soundfile as sf
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict

logger = logging.getLogger(__name__)

class NoiseReductionBase(ABC):
    """Lightweight base class for noise reduction models"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.is_ready = False
        
    @abstractmethod
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """Enhance audio by removing noise"""
        pass
    
    def resample_if_needed(self, audio: np.ndarray, input_sr: int) -> tuple[np.ndarray, bool]:
        """Resample only if necessary, return audio and whether resampling occurred"""
        if input_sr == self.sample_rate:
            return audio, False
        
        # Use the same device as the model for resampling
        device = getattr(self, 'device', torch.device("cpu"))
        
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)
        resampled = torchaudio.functional.resample(audio_tensor, input_sr, self.sample_rate)
        return resampled.squeeze().cpu().numpy(), True


class DeepFilterNet3(NoiseReductionBase):
    """DeepFilterNet3 - Best balance of quality and speed for real-time"""
    
    def __init__(self):
        super().__init__(sample_rate=48000)
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            from df.enhance import init_df, enhance
            from df.io import load_audio, save_audio
            
            self.model, self.df_state, _ = init_df(config_allow_defaults=True)
            self.device = next(self.model.parameters()).device
            self.model.eval()
            
            # Cache the functions to avoid repeated imports
            self.enhance_fn = enhance
            self.load_audio_fn = load_audio
            self.save_audio_fn = save_audio
            
            self.is_ready = True
            logger.info(f"DeepFilterNet3 initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize DeepFilterNet3: {e}")
            raise
    
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            raise RuntimeError("Model not ready")
        
        # Use temporary file for DeepFilterNet (required by its API)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            try:
                sf.write(temp_file.name, audio, self.sample_rate)
                sample, _ = self.load_audio_fn(temp_file.name, self.sample_rate)
                
                # Convert to mono if stereo
                if sample.dim() > 1 and sample.shape[0] > 1:
                    sample = sample.mean(dim=0, keepdim=True)
                
                enhanced = self.enhance_fn(self.model, self.df_state, sample)
                self.save_audio_fn(temp_file.name, enhanced, self.sample_rate)
                enhanced_audio, _ = sf.read(temp_file.name, dtype='float32')
                
                return enhanced_audio
            finally:
                os.unlink(temp_file.name)


class ClearVoiceSpeechEnhancement(NoiseReductionBase):
    """ClearVoice Speech Enhancement - High quality real-time enhancement"""
    
    def __init__(self, model_name: str = "MossFormer2_SE_48K"):
        sample_rate = 48000 if "48K" in model_name else 16000
        super().__init__(sample_rate=sample_rate)
        self.model_name = model_name
        self.device = None
        
        self.supported_models = {
            "FRCRN_SE_16K": "speech_enhancement",
            "MossFormer2_SE_48K": "speech_enhancement", 
            "MossFormerGAN_SE_16K": "speech_enhancement"
        }
        
        if model_name not in self.supported_models:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            from clearvoice import ClearVoice
            
            # Let ClearVoice choose its preferred device
            self.clearvoice = ClearVoice(task='speech_enhancement', model_names=[self.model_name])
            
            # Try to detect the device ClearVoice is actually using
            self._detect_model_device()
            
            self.is_ready = True
            logger.info(f"ClearVoice {self.model_name} initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_name}: {e}")
            raise
    
    def _detect_model_device(self):
        """Detect what device the ClearVoice model is actually using"""
        try:
            # Try to access the model's device through ClearVoice internals
            if hasattr(self.clearvoice, 'model') and hasattr(self.clearvoice.model, 'parameters'):
                self.device = next(self.clearvoice.model.parameters()).device
            elif hasattr(self.clearvoice, 'models') and self.clearvoice.models:
                # Some ClearVoice versions store models in a dict
                first_model = next(iter(self.clearvoice.models.values()))
                if hasattr(first_model, 'parameters'):
                    self.device = next(first_model.parameters()).device
            else:
                # Fallback: assume it's using the default device
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
            logger.info(f"Detected ClearVoice device: {self.device}")
        except Exception as e:
            logger.warning(f"Could not detect device for {self.model_name}, using default: {e}")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            raise RuntimeError(f"Model {self.model_name} not ready")
        
        # Normalize input
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
            scale_factor = max_val
        else:
            scale_factor = 1.0
        
        # Process with ClearVoice (let it handle its own device management)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            try:
                sf.write(tmp_file.name, audio, self.sample_rate)
                enhanced_dict = self.clearvoice(input_path=tmp_file.name, online_write=False)
                
                # Extract enhanced audio
                enhanced_audio = next(iter(enhanced_dict.values())) if isinstance(enhanced_dict, dict) else enhanced_dict
                
                if torch.is_tensor(enhanced_audio):
                    # Move to CPU for output (let ClearVoice handle its own device)
                    enhanced_audio = enhanced_audio.cpu().numpy()
                
                # Restore original scale
                enhanced_audio = enhanced_audio * scale_factor
                return enhanced_audio.astype(np.float32)
                
            finally:
                os.unlink(tmp_file.name)


class ClearVoiceSpeechSeparation(NoiseReductionBase):
    """ClearVoice Speech Separation - Multi-speaker separation"""
    
    def __init__(self, model_name: str = "MossFormer2_SS_16K"):
        super().__init__(sample_rate=16000)
        self.model_name = model_name
        self.device = None
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            from clearvoice import ClearVoice
            
            self.clearvoice = ClearVoice(task='speech_separation', model_names=[self.model_name])
            
            # Detect the actual device ClearVoice is using
            self._detect_model_device()
                
            self.is_ready = True
            logger.info(f"ClearVoice {self.model_name} initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_name}: {e}")
            raise
    
    def _detect_model_device(self):
        """Detect what device the ClearVoice model is actually using"""
        try:
            if hasattr(self.clearvoice, 'model') and hasattr(self.clearvoice.model, 'parameters'):
                self.device = next(self.clearvoice.model.parameters()).device
            elif hasattr(self.clearvoice, 'models') and self.clearvoice.models:
                first_model = next(iter(self.clearvoice.models.values()))
                if hasattr(first_model, 'parameters'):
                    self.device = next(first_model.parameters()).device
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
            logger.info(f"Detected ClearVoice device: {self.device}")
        except Exception as e:
            logger.warning(f"Could not detect device for {self.model_name}, using default: {e}")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """For FastRTC integration, return the first separated speaker"""
        if not self.is_ready:
            raise RuntimeError(f"Model {self.model_name} not ready")
        
        # Normalize input
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
            scale_factor = max_val
        else:
            scale_factor = 1.0
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            try:
                sf.write(tmp_file.name, audio, self.sample_rate)
                separated_dict = self.clearvoice(input_path=tmp_file.name, online_write=False)
                
                # Get the first speaker for real-time streaming
                if isinstance(separated_dict, dict):
                    first_speaker = next(iter(separated_dict.values()))
                else:
                    first_speaker = separated_dict
                
                if torch.is_tensor(first_speaker):
                    first_speaker = first_speaker.cpu().numpy()
                
                # Restore original scale
                first_speaker = first_speaker * scale_factor
                return first_speaker.astype(np.float32)
                
            finally:
                os.unlink(tmp_file.name)


class SpectralSubtraction(NoiseReductionBase):
    """Fast spectral subtraction - Ultra low latency fallback"""
    
    def __init__(self, alpha: float = 2.0, beta: float = 0.01):
        super().__init__(sample_rate=48000)  # Match FastRTC default
        self.alpha = alpha
        self.beta = beta
        self.noise_profile = None
        self.is_ready = True
        logger.info("Spectral Subtraction initialized")
    
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        # Estimate noise profile from first 0.5 seconds if not available
        if self.noise_profile is None:
            noise_samples = min(len(audio), self.sample_rate // 2)
            self.noise_profile = np.abs(np.fft.fft(audio[:noise_samples]))
        
        audio_fft = np.fft.fft(audio)
        magnitude = np.abs(audio_fft)
        phase = np.angle(audio_fft)
        
        # Spectral subtraction
        enhanced_magnitude = magnitude - self.alpha * self.noise_profile[:len(magnitude)]
        enhanced_magnitude = np.maximum(enhanced_magnitude, self.beta * magnitude)
        
        enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = np.real(np.fft.ifft(enhanced_fft))
        
        return enhanced_audio.astype(np.float32)


class FastRTCNoiseReduction:
    """
    Optimized noise reduction wrapper for FastRTC integration
    Loads model first and adapts to the device the model actually uses
    """
    
    def __init__(self, preferred_model: str = "deepfilternet3"):
        self.model = None
        self.model_name = None
        self.device = None
        self._initialize_model(preferred_model)
    
    def _initialize_model(self, preferred_model: str):
        """Initialize the exact model requested and detect its device"""
        models_to_try = {
            "mossformer2_se_48k": lambda: ClearVoiceSpeechEnhancement("MossFormer2_SE_48K"),
            "frcrn_se_16k": lambda: ClearVoiceSpeechEnhancement("FRCRN_SE_16K"),
            "mossformergan_se_16k": lambda: ClearVoiceSpeechEnhancement("MossFormerGAN_SE_16K"),
            "deepfilternet3": DeepFilterNet3
        }
        
        if preferred_model not in models_to_try:
            raise ValueError(f"Unsupported model: {preferred_model}. Supported: {list(models_to_try.keys())}")
        
        try:
            logger.info(f"Initializing {preferred_model}...")
            
            # Load the model and let it choose its device
            if callable(models_to_try[preferred_model]):
                self.model = models_to_try[preferred_model]()
            else:
                self.model = models_to_try[preferred_model]()
                    
            if self.model.is_ready:
                self.model_name = preferred_model
                # Get the device the model is actually using
                self.device = getattr(self.model, 'device', torch.device("cpu"))
                logger.info(f"Successfully initialized {preferred_model} on {self.device}")
            else:
                raise RuntimeError(f"Model {preferred_model} failed to become ready")
                    
        except Exception as e:
            logger.error(f"Failed to initialize {preferred_model}: {e}")
            raise
    
    def process_audio_chunk(self, audio: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
        """
        Process a single audio chunk - optimized for FastRTC streaming
        
        Args:
            audio: Audio data as numpy array (shape: [samples] or [channels, samples])
            sample_rate: Input sample rate
            
        Returns:
            Enhanced audio as numpy array
        """
        if self.model is None:
            return audio
        
        # Handle stereo input - convert to mono for processing
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # Normalize input to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
            need_rescale = True
            scale_factor = max_val
        else:
            need_rescale = False
            scale_factor = 1.0
        
        # Resample if needed (using model's detected device)
        audio_for_model, was_resampled = self.model.resample_if_needed(audio, sample_rate)
        
        # Enhance audio (model handles its own device management)
        enhanced_audio = self.model.enhance_audio(audio_for_model)
        
        # Resample back if needed (using same device as model)
        if was_resampled:
            model_device = getattr(self.model, 'device', torch.device("cpu"))
            enhanced_tensor = torch.from_numpy(enhanced_audio).float().unsqueeze(0).to(model_device)
            enhanced_audio = torchaudio.functional.resample(
                enhanced_tensor, self.model.sample_rate, sample_rate
            ).squeeze().cpu().numpy()
        
        # Restore original scale if needed
        if need_rescale:
            enhanced_audio = enhanced_audio * scale_factor
        
        return enhanced_audio.astype(np.float32)
    
    def get_model_info(self) -> dict:
        """Get current model information"""
        return {
            "model_name": self.model_name,
            "sample_rate": self.model.sample_rate if self.model else None,
            "is_ready": self.model.is_ready if self.model else False,
            "device": getattr(self.model, 'device', 'cpu')
        }


# FastRTC Integration Helper
class AudioNoiseReductionHandler:
    """
    Example FastRTC StreamHandler with noise reduction
    Use this as a template for your FastRTC integration
    """
    
    def __init__(self, 
                 noise_reduction_model: str = "mossformer2_se_48k",
                 expected_layout: str = "mono", 
                 output_sample_rate: int = 48000):
        self.noise_reducer = FastRTCNoiseReduction(noise_reduction_model)
        self.expected_layout = expected_layout
        self.output_sample_rate = output_sample_rate
        
        logger.info(f"Audio handler initialized with {self.noise_reducer.get_model_info()}")
    
    def process_audio_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """
        Process a single audio frame from FastRTC
        This method should be called from your FastRTC handler
        """
        try:
            # Apply noise reduction
            enhanced_audio = self.noise_reducer.process_audio_chunk(
                audio_frame, self.output_sample_rate
            )
            return enhanced_audio
            
        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            # Return original audio on error to maintain stream continuity
            return audio_frame


# Simple test function
def test_noise_reduction():
    """Test the noise reduction system"""
    print("Testing FastRTC Noise Reduction...")
    
    # Create test audio (1 second at 48kHz)
    sample_rate = 48000
    duration = 1.0
    test_audio = np.random.normal(0, 0.1, int(sample_rate * duration)).astype(np.float32)
    
    models_to_test = [
        "mossformer2_se_48k", 
        "frcrn_se_16k", 
        "mossformergan_se_16k",
        "mossformer2_ss_16k",
        "deepfilternet3", 
        "spectral"
    ]
    
    for model_name in models_to_test:
        try:
            print(f"\nTesting {model_name}...")
            noise_reducer = FastRTCNoiseReduction(model_name)
            enhanced_audio = noise_reducer.process_audio_chunk(test_audio, sample_rate)
            
            info = noise_reducer.get_model_info()
            print(f"✅ Success with {info['model_name']}")
            print(f"   Input: {len(test_audio)} samples")
            print(f"   Output: {len(enhanced_audio)} samples")
            print(f"   Sample rate: {info['sample_rate']}Hz")
            
        except Exception as e:
            print(f"❌ {model_name}: Error - {e}")
    
    # Test adaptive fallback
    try:
        print(f"\nTesting adaptive fallback...")
        noise_reducer = FastRTCNoiseReduction()
        enhanced_audio = noise_reducer.process_audio_chunk(test_audio, sample_rate)
        info = noise_reducer.get_model_info()
        print(f"✅ Adaptive success with {info['model_name']}")
        
    except Exception as e:
        print(f"❌ Adaptive: Error - {e}")


if __name__ == "__main__":
    test_noise_reduction()