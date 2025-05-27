"""
Hugging Face Speech-to-Text model implementation.

This module provides a wrapper around Transformers-based STT models from Hugging Face,
conforming to the BaseSTTModel interface.
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, Iterator, Tuple, List

import warnings
warnings.filterwarnings("ignore", message="The input name `inputs` is deprecated")
warnings.filterwarnings("ignore", message="You have passed task=transcribe")

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import BitsAndBytesConfig

from .stt_base_model import BaseSTTModel, STTConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HFSTTConfig(STTConfig):
    """Configuration for Hugging Face STT model.
    
    Attributes:
        checkpoint: Model checkpoint name or path
        model_folder_path: Path to folder containing model files
        batch_size: Batch size for inference
        cuda_device_id: CUDA device ID for GPU acceleration
        chunk_length_s: Chunk length in seconds for processing long audio
        compute_type: Compute type for model inference (float16, int8)
        beam_size: Beam search size for decoding
        show_logs: Whether to display log messages
    """
    def __init__(self, **kwargs):
        """Initialize HF STT configuration with defaults."""
        super().__init__(**kwargs)
        
        # Model identification
        self.checkpoint = kwargs.get("checkpoint", "whisper-large-v3-turbo")
        self.model_folder_path = kwargs.get("model_folder_path", "../../HF_models")
        
        # Performance settings
        self.batch_size = kwargs.get("batch_size", 8)
        self.cuda_device_id = kwargs.get("cuda_device_id", 0)
        self.chunk_length_s = kwargs.get("chunk_length_s", 30)
        self.compute_type = kwargs.get("compute_type", "float16")
        self.beam_size = kwargs.get("beam_size", 5)
        
        # Use 4-bit quantization by default
        self.use_4bit = kwargs.get("use_4bit", True)


class HFSTTModel(BaseSTTModel):
    """Implementation of Hugging Face Transformers STT model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Hugging Face STT model with configuration settings.
        
        Args:
            config: Dictionary with STT configuration options.
        """
        # Get configuration or use defaults
        config = config or {}
            
        # Create STT configuration
        self.config = HFSTTConfig(**config)
        
        # Configure logging based on show_logs
        self.configure_logging()
        
        # Setup device
        self._setup_device()
        
        # Load model and processor
        self._load_model()
        
    def configure_logging(self) -> None:
        """Configure logging based on show_logs setting."""
        if not self.config.show_logs:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
    
    def _setup_device(self) -> None:
        """Set up device for model execution."""
        # Setup device
        self.device = f"cuda:{self.config.cuda_device_id}" if torch.cuda.is_available() else "cpu"
        
        # Set compute type
        if self.config.compute_type == "float16" and torch.cuda.is_available():
            self.torch_dtype = torch.float16
        elif self.config.compute_type == "int8":
            self.torch_dtype = torch.int8
        else:
            self.torch_dtype = torch.float32
            
        # Print device info if logs enabled
        if self.config.show_logs:
            print(f"Device set to use {self.device}")
    
    def _load_model(self) -> None:
        """Load model and processor from checkpoint."""
        # Determine model path
        self.model_path = os.path.join(self.config.model_folder_path, self.config.checkpoint)
        if not os.path.exists(self.model_path) and not self.config.checkpoint.startswith(("http://", "https://")):
            if self.config.show_logs:
                logger.info(f"Model path {self.model_path} not found, using checkpoint name directly: {self.config.checkpoint}")
            self.model_path = self.config.checkpoint

        # Configure quantization if enabled
        quantization_config = None
        if self.config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        
        if self.config.show_logs:
            logger.info(f"Loading STT model from {self.model_path} on {self.device}")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            quantization_config=quantization_config,
            use_safetensors=True,
            device_map=self.device
        )

        # Create pipeline without generate_kwargs to avoid forced_decoder_ids issue
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,            
            torch_dtype=self.torch_dtype,
            chunk_length_s=self.config.chunk_length_s
        )
        
        # Set num_beams directly on the model's generation_config instead
        if hasattr(self.model, "generation_config") and self.config.beam_size > 0:
            self.model.generation_config.num_beams = self.config.beam_size
        
        if self.config.show_logs:
            logger.info(f"STT model {self.config.checkpoint} loaded successfully")
        
    def update_config(self, **kwargs) -> None:
        """
        Update STT configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        # Track changes that require model reloading
        requires_reload = False
        old_show_logs = self.config.show_logs
        
        # Update configuration attributes
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                # Check if this parameter requires model reload
                if key in ["checkpoint", "model_folder_path", "compute_type", "use_4bit", "cuda_device_id"]:
                    requires_reload = True
                
                # Update the configuration
                setattr(self.config, key, value)
                
                if self.config.show_logs:
                    logger.debug(f"Updated config {key} = {value}")
            else:
                if self.config.show_logs:
                    logger.warning(f"Ignoring unknown config parameter: {key}")
        
        # Update logging if show_logs changed
        if old_show_logs != self.config.show_logs:
            self.configure_logging()
            
        # Update pipeline settings that don't require reload
        if not requires_reload and hasattr(self, "pipe"):
            if "chunk_length_s" in kwargs:
                self.pipe.chunk_length_s = self.config.chunk_length_s
            
            if "beam_size" in kwargs and hasattr(self.pipe, "model"):
                self.pipe.generate_kwargs["num_beams"] = self.config.beam_size
                
            if "batch_size" in kwargs:
                # Can't update batch_size directly in pipeline
                pass
        
        # Reload model if necessary
        if requires_reload:
            if self.config.show_logs:
                logger.info("Reloading model due to configuration changes")
            self._setup_device()
            self._load_model()
            
        if self.config.show_logs:
            logger.info(f"Updated HF STT configuration: {self.config}")
    
    def _process_audio_input(self, audio_data: Any) -> np.ndarray:
        """
        Process audio data into the format expected by the model.
        
        Args:
            audio_data: Audio data to process, can be:
                - File path (str)
                - Audio array (numpy.ndarray)
                - Tuple of (sample_rate, audio_array)
                
        Returns:
            Processed audio array
            
        Raises:
            ValueError: If audio data format is not supported
        """
        if audio_data is None:
            raise ValueError("No audio data provided")
        
        # Handle file path
        if isinstance(audio_data, str):
            return audio_data
        
        # Handle tuple from WebRTC component (sample_rate, audio_array)
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            _, audio_array = audio_data
            
            if isinstance(audio_array, np.ndarray):
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.squeeze()
                
                return audio_array.astype("float32") / 32768.0
            else:
                raise ValueError(f"Unsupported audio array type: {type(audio_array)}")
        
        # Direct numpy array input
        if isinstance(audio_data, np.ndarray):
            if len(audio_data.shape) > 1:
                audio_array = audio_data.squeeze()
            else:
                audio_array = audio_data
            
            return audio_array.astype("float32") / 32768.0
        
        # Unsupported type
        raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
    
    def transcribe(self, audio_data: Any, task: str = "transcribe") -> Union[str, Dict[str, Any]]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio data to transcribe, can be:
                - File path (str)
                - Audio array (numpy.ndarray)
                - Tuple of (sample_rate, audio_array)
            task: Either 'transcribe' (same language) or 'translate' (to English)
            
        Returns:
            Transcribed text or dictionary with text and metadata
            
        Raises:
            ValueError: If audio data format is not supported
        """
        try:
            # Process audio into the right format
            processed_audio = self._process_audio_input(audio_data)
            
            # Run transcription without any generate_kwargs to avoid forced_decoder_ids issue
            result = self.pipe(processed_audio, batch_size=self.config.batch_size)
            
            return result
            
        except Exception as e:
            if self.config.show_logs:
                logger.error(f"Error in transcribe: {e}")
            raise ValueError(f"Transcription error: {str(e)}")
            
    def transcribe_stream(self, audio_stream, task: str = "transcribe") -> Iterator[Union[str, Dict[str, Any]]]:
        """
        Transcribe streaming audio data.
        
        Note: The current HF implementation doesn't support true streaming.
        This method buffers chunks and processes them in batches.
        
        Args:
            audio_stream: Stream of audio chunks
            task: Either 'transcribe' (same language) or 'translate' (to English)
            
        Yields:
            Transcribed text segments as they become available
        """
        if self.config.show_logs:
            logger.warning("Stream transcription not fully implemented for HF models")
        
        buffer: List[np.ndarray] = []
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

