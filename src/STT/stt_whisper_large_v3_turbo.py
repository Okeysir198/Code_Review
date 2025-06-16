## src/STT/stt_whisper_large_v3_turbo.py
"""
Optimized Hugging Face Whisper model with best-practice configurations.
Fixed based on working implementation patterns.
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, Iterator
import warnings
from dataclasses import dataclass

# Suppress model warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message="The input name `inputs` is deprecated")
warnings.filterwarnings("ignore", message="You have passed task=transcribe")

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import BitsAndBytesConfig

from .stt_base_model import BaseSTTModel, STTConfig

logger = logging.getLogger(__name__)

@dataclass  
class OPENAIWhisperV3TurboConfig(STTConfig):
    """Optimized configuration for Whisper models."""
    
    # Model settings - optimized for whisper-large-v3-turbo
    checkpoint: str = "openai/whisper-large-v3-turbo"
    model_folder_path: str = "./models"
    
    # Performance optimizations
    batch_size: int = 4  # Optimal balance of speed/memory
    chunk_length_s: int = 30  # Whisper's native chunk size
    compute_type: str = "float16"  # Optimal for GPU inference
    beam_size: int = 1  # Greedy decoding for speed (RTF optimization)
    
    # Memory optimizations 
    use_4bit: bool = False  # Disabled for better quality
    low_cpu_mem_usage: bool = True
    use_safetensors: bool = True
    
    # Audio processing optimizations
    return_timestamps: bool = True
    condition_on_prev_tokens: bool = False  # Reduces hallucinations
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6


class OPENAIWhisperV3TurboModel(BaseSTTModel):
    """Optimized Whisper implementation with best practices."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optimized settings."""
        # Call parent first to set up _raw_config
        super().__init__(config)
        
        # Then create model-specific config
        self.config = OPENAIWhisperV3TurboConfig(**(self._raw_config or {}))
        
        # Configure logging based on show_logs
        self.configure_logging()
        
        self._setup_device()
        self._load_model()
        
    def configure_logging(self) -> None:
        """Configure logging based on show_logs setting."""
        if not self.config.show_logs:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
    
    def _setup_device(self) -> None:
        """Setup optimal device configuration."""
        if torch.cuda.is_available() and self.config.use_gpu:
            self.device = f"cuda:{self.config.cuda_device_id}"
            
            # Set compute type based on config
            if self.config.compute_type == "float16":
                self.torch_dtype = torch.float16
            elif self.config.compute_type == "int8":
                self.torch_dtype = torch.int8
            else:
                self.torch_dtype = torch.float32
        else:
            self.device = "cpu" 
            self.torch_dtype = torch.float32
            
        if self.config.show_logs:
            logger.info(f"Using device: {self.device}, dtype: {self.torch_dtype}")
    
    def _load_model(self) -> None:
        """Load model with optimized configuration - Based on working implementation."""
        try:
            # Determine model path
            model_path = os.path.join(self.config.model_folder_path, self.config.checkpoint)
            if not os.path.exists(model_path) and not self.config.checkpoint.startswith(("http://", "https://")):
                if self.config.show_logs:
                    logger.info(f"Model path {model_path} not found, using checkpoint name directly: {self.config.checkpoint}")
                model_path = self.config.checkpoint

            # Configure quantization if enabled
            quantization_config = None
            if self.config.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            
            if self.config.show_logs:
                logger.info(f"Loading STT model from {model_path} on {self.device}")
            
            # Load processor first
            self.processor = AutoProcessor.from_pretrained(model_path)
            
            # Load model with working configuration pattern
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path, 
                torch_dtype=self.torch_dtype, 
                low_cpu_mem_usage=self.config.low_cpu_mem_usage, 
                quantization_config=quantization_config,
                use_safetensors=self.config.use_safetensors,
                device_map=self.device  # Use device_map like working version
            )

            # Create pipeline without complex parameters - follow working pattern
            self.pipe = pipeline(
                task="automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,            
                torch_dtype=self.torch_dtype,
                chunk_length_s=self.config.chunk_length_s
                # Don't specify device - let device_map handle it
            )
            
            # Set num_beams directly on the model's generation_config instead
            if hasattr(self.model, "generation_config") and self.config.beam_size > 0:
                self.model.generation_config.num_beams = self.config.beam_size
            
            if self.config.show_logs:
                logger.info(f"STT model {self.config.checkpoint} loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _process_audio_input(self, audio_data: Any) -> Union[str, np.ndarray]:
        """Process audio with optimal preprocessing - Based on working implementation."""
        if audio_data is None:
            raise ValueError("No audio data provided")
        
        # Handle file path
        if isinstance(audio_data, str):
            return audio_data
        
        # Handle WebRTC tuple format (sample_rate, audio_array)
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            _, audio_array = audio_data
            
            if isinstance(audio_array, np.ndarray):
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.squeeze()
                
                # Normalize by max absolute value
                max_val = np.max(np.abs(audio_array))
                if max_val > 0:  # Avoid division by zero
                    return audio_array.astype("float32") / max_val
                else:
                    return audio_array.astype("float32")  # All zeros
            else:
                raise ValueError(f"Unsupported audio array type: {type(audio_array)}")
        
        # Handle direct numpy array
        if isinstance(audio_data, np.ndarray):
            if len(audio_data.shape) > 1:
                audio_array = audio_data.squeeze()
            else:
                audio_array = audio_data
            
            # Normalize by max absolute value
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:  # Avoid division by zero
                return audio_array.astype("float32") / max_val
            else:
                return audio_array.astype("float32")  # All zeros
        
        raise ValueError(f"Unsupported audio format: {type(audio_data)}")
    
    def transcribe(self, audio_data: Any, task: str = "translate") -> Union[str, Dict[str, Any]]:
        """Transcribe with optimal settings - Based on working implementation."""
        try:
            # Process audio into the right format
            processed_audio = self._process_audio_input(audio_data)
            
            # Run transcription without any generate_kwargs to avoid forced_decoder_ids issue
            result = self.pipe(processed_audio, 
                               batch_size=self.config.batch_size,
                               generate_kwargs={"language": "english", "task": task}, #translate, transcribe
                               )
            
            return result
            
        except Exception as e:
            if self.config.show_logs:
                logger.error(f"Error in transcribe: {e}")
            raise ValueError(f"Transcription error: {str(e)}")
    
    def transcribe_stream(self, audio_stream, task: str = "transcribe") -> Iterator[Union[str, Dict[str, Any]]]:
        """Optimized streaming transcription - Based on working implementation."""
        if self.config.show_logs:
            logger.warning("Stream transcription not fully implemented for HF models")
        
        buffer = []
        buffer_duration_seconds = 0
        buffer_threshold_seconds = min(10, self.config.chunk_length_s)  # Process at shorter intervals
        
        try:
            # Process audio chunks as they come
            for chunk in audio_stream:
                # Process the chunk to get sample rate and numpy array
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    sample_rate, audio_array = chunk
                    
                    # Skip empty chunks
                    if len(audio_array) == 0:
                        continue
                    
                    # Normalize and convert format
                    if isinstance(audio_array, np.ndarray):
                        if len(audio_array.shape) > 1:
                            audio_array = audio_array.squeeze()
                        audio_array = audio_array.astype("float32") / 32768.0
                    
                    # Add to buffer
                    buffer.append(audio_array)
                    
                    # Update buffer duration
                    chunk_duration = len(audio_array) / sample_rate
                    buffer_duration_seconds += chunk_duration
                    
                    # Process buffer when it reaches threshold
                    if buffer_duration_seconds >= buffer_threshold_seconds:
                        # Combine chunks
                        combined = np.concatenate(buffer)
                        
                        # Transcribe without any generate_kwargs to avoid forced_decoder_ids issue
                        result = self.pipe(combined, batch_size=self.config.batch_size)
                        
                        # Yield the transcription
                        yield result["text"]
                        
                        # Clear buffer
                        buffer = []
                        buffer_duration_seconds = 0
            
            # Process any remaining audio in buffer
            if buffer:
                combined = np.concatenate(buffer)
                
                # Transcribe without any generate_kwargs to avoid forced_decoder_ids issue
                result = self.pipe(combined, batch_size=self.config.batch_size)
                    
                yield result["text"]
                
        except Exception as e:
            if self.config.show_logs:
                logger.error(f"Error in stream transcription: {e}")
            yield f"Transcription error: {str(e)}"
    