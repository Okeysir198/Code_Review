"""
Real-time Noise Cancellation for FastRTC Integration
Optimized for minimum latency and maximum performance with direct tensor processing.
"""

import numpy as np
import torch
import torchaudio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Tuple
import warnings

logger = logging.getLogger(__name__)

class NoiseReductionBase(ABC):
    """Lightweight base class for noise reduction models"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.is_ready = False
        self.device = torch.device("cpu")
        
    @abstractmethod
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """Enhance audio by removing noise"""
        pass
    
    def resample_if_needed(self, audio: np.ndarray, input_sr: int) -> Tuple[np.ndarray, bool]:
        """Resample only if necessary, return audio and whether resampling occurred"""
        if input_sr == self.sample_rate:
            return audio, False
        
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
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
            
            # Initialize model and state
            self.model, self.df_state, _ = init_df(config_allow_defaults=True)
            self.device = next(self.model.parameters()).device
            self.model.eval()
            
            # Cache the enhance function
            self.enhance_fn = enhance
            
            self.is_ready = True
            logger.info(f"DeepFilterNet3 initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize DeepFilterNet3: {e}")
            raise
    
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            raise RuntimeError("Model not ready")
        
        # Convert to torch tensor with proper shape [C, T] 
        # Keep on CPU as DeepFilterNet STFT operations require CPU
        if audio.ndim == 1:
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # [1, T]
        else:
            audio_tensor = torch.from_numpy(audio).float()
        
        # DeepFilterNet enhance function handles device management internally
        # It keeps STFT on CPU but moves neural network parts to GPU
        with torch.no_grad():
            enhanced_tensor = self.enhance_fn(self.model, self.df_state, audio_tensor)
        
        # Convert result to numpy - enhanced_tensor should be on CPU already
        if torch.is_tensor(enhanced_tensor):
            # Should already be on CPU, but ensure it
            if enhanced_tensor.is_cuda:
                enhanced_audio = enhanced_tensor.cpu().numpy()
            else:
                enhanced_audio = enhanced_tensor.numpy()
        else:
            # Already numpy
            enhanced_audio = enhanced_tensor
        
        # Handle different output shapes
        if enhanced_audio.ndim > 1:
            enhanced_audio = enhanced_audio.squeeze()
            
        return enhanced_audio.astype(np.float32)


class ClearVoiceSpeechEnhancement(NoiseReductionBase):
    """ClearVoice Speech Enhancement - High quality real-time enhancement"""
    
    def __init__(self, model_name: str = "MossFormer2_SE_48K"):
        sample_rate = 48000 if "48K" in model_name else 16000
        super().__init__(sample_rate=sample_rate)
        self.model_name = model_name
        
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
            from clearvoice.utils.decode import decode_one_audio
            
            # Initialize ClearVoice to get the model and args
            self.clearvoice = ClearVoice(task='speech_enhancement', model_names=[self.model_name])
            
            # Extract the actual model and arguments for direct processing
            # ClearVoice stores models in a list
            self.network = self.clearvoice.models[0]  # Get the first (and only) model
            self.model = self.network.model  # The actual torch model
            self.args = self.network.args    # The arguments object
            
            # Detect device
            self._detect_model_device()
            
            # Cache the decode function
            self.decode_fn = decode_one_audio
            
            self.is_ready = True
            logger.info(f"ClearVoice {self.model_name} initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_name}: {e}")
            raise
    
    def _detect_model_device(self):
        """Detect what device the ClearVoice model is actually using"""
        try:
            if hasattr(self.model, 'parameters'):
                self.device = next(self.model.parameters()).device
            elif hasattr(self.network, 'device'):
                self.device = self.network.device
            else:
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
        
        # Prepare input tensor [B, T] format
        if audio.ndim == 1:
            inputs = audio[np.newaxis, :]  # Add batch dimension [1, T]
        else:
            inputs = audio
        
        # Use ClearVoice's direct decode function
        with torch.no_grad():
            enhanced_audio = self.decode_fn(self.model, self.network.device, inputs, self.args)
        
        # Handle different output formats
        if isinstance(enhanced_audio, (list, tuple)):
            enhanced_audio = enhanced_audio[0]  # Take first output
        
        if torch.is_tensor(enhanced_audio):
            enhanced_audio = enhanced_audio.cpu().numpy()
        
        # Restore original scale
        enhanced_audio = enhanced_audio * scale_factor
        return enhanced_audio.astype(np.float32)


class ClearVoiceSpeechSeparation(NoiseReductionBase):
    """ClearVoice Speech Separation - Multi-speaker separation"""
    
    def __init__(self, model_name: str = "MossFormer2_SS_16K"):
        super().__init__(sample_rate=16000)
        self.model_name = model_name
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            from clearvoice import ClearVoice
            from clearvoice.utils.decode import decode_one_audio
            
            self.clearvoice = ClearVoice(task='speech_separation', model_names=[self.model_name])
            
            # Extract model and args for direct processing
            # ClearVoice stores models in a list
            self.network = self.clearvoice.models[0]  # Get the first (and only) model
            self.model = self.network.model  # The actual torch model
            self.args = self.network.args    # The arguments object
            
            # Detect device
            self._detect_model_device()
            
            # Cache decode function
            self.decode_fn = decode_one_audio
                
            self.is_ready = True
            logger.info(f"ClearVoice {self.model_name} initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_name}: {e}")
            raise
    
    def _detect_model_device(self):
        """Detect what device the ClearVoice model is actually using"""
        try:
            if hasattr(self.model, 'parameters'):
                self.device = next(self.model.parameters()).device
            elif hasattr(self.network, 'device'):
                self.device = self.network.device
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
        
        # Prepare input tensor [B, T] format
        if audio.ndim == 1:
            inputs = audio[np.newaxis, :]  # Add batch dimension [1, T]
        else:
            inputs = audio
        
        # Use ClearVoice's direct decode function
        with torch.no_grad():
            separated_outputs = self.decode_fn(self.model, self.network.device, inputs, self.args)
        
        # Get the first speaker for real-time streaming
        if isinstance(separated_outputs, (list, tuple)):
            first_speaker = separated_outputs[0]
        else:
            first_speaker = separated_outputs
        
        if torch.is_tensor(first_speaker):
            first_speaker = first_speaker.cpu().numpy()
        
        # Restore original scale
        first_speaker = first_speaker * scale_factor
        return first_speaker.astype(np.float32)




class FastRTCNoiseReduction:
    """
    Optimized noise reduction wrapper for FastRTC integration
    Uses direct tensor processing for minimal latency
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
            "mossformer2_ss_16k": lambda: ClearVoiceSpeechSeparation("MossFormer2_SS_16K"),
            "deepfilternet3": DeepFilterNet3,
        }
        
        if preferred_model not in models_to_try:
            raise ValueError(f"Unsupported model: {preferred_model}. Supported: {list(models_to_try.keys())}")
        
        try:
            logger.info(f"Initializing {preferred_model}...")
            
            # Load the model
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
    
    def process_audio_chunk(self, audio: np.ndarray, input_sample_rate: int ) -> np.ndarray:
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
        
        # Ensure audio is contiguous and float32
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        
        # Normalize input to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val
            need_rescale = True
            scale_factor = max_val
        else:
            need_rescale = False
            scale_factor = 1.0
        
        # Resample if needed
        audio_for_model, was_resampled = self.model.resample_if_needed(audio, input_sample_rate)
        # logger.info(f"Audio was resampled: {was_resampled}. From sr:{input_sample_rate} Hz to sr: {self.model.sample_rate} Hz")

        # Enhance audio using direct tensor processing
        enhanced_audio = self.model.enhance_audio(audio_for_model)
        
        # Resample back if needed
        if was_resampled:
            model_device = getattr(self.model, 'device', torch.device("cpu"))
            enhanced_tensor = torch.from_numpy(enhanced_audio).float().unsqueeze(0).to(model_device)
            enhanced_audio = torchaudio.functional.resample(
                enhanced_tensor, self.model.sample_rate, input_sample_rate
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
            "device": str(getattr(self.model, 'device', 'cpu'))
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
            # Apply noise reduction using direct tensor processing
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
    print("Testing FastRTC Noise Reduction (Direct Tensor Processing)...")
    
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
            print(f"   Device: {info['device']}")
            
        except Exception as e:
            print(f"❌ {model_name}: Error - {e}")
    
    print(f"\n🚀 All tests completed!")


if __name__ == "__main__":
    test_noise_reduction()