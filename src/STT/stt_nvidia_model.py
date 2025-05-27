"""
NVIDIA Parakeet-TDT Speech-to-Text model implementation.

This module provides a wrapper around NVIDIA's Parakeet-TDT models from NeMo framework,
conforming to the BaseSTTModel interface.
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, Union, Iterator, List

try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

from .stt_base_model import BaseSTTModel, STTConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NVIDIAParakeetConfig(STTConfig):
    """Configuration for NVIDIA Parakeet-TDT STT model."""
    
    def __init__(self, **kwargs):
        """Initialize NVIDIA Parakeet-TDT configuration with defaults."""
        super().__init__(**kwargs)
        
        # Model identification
        self.checkpoint = kwargs.get("checkpoint", "nvidia/parakeet-tdt-0.6b-v2")
        self.model_folder_path = kwargs.get("model_folder_path", "./models")
        
        # Performance settings
        self.batch_size = kwargs.get("batch_size", 1)
        self.cuda_device_id = kwargs.get("cuda_device_id", 0)
        self.chunk_length_s = kwargs.get("chunk_length_s", 60)


class NVIDIAParakeetModel(BaseSTTModel):
    """Implementation of NVIDIA Parakeet-TDT model for speech recognition."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the NVIDIA Parakeet-TDT model with configuration settings."""
        # Check if NeMo is available
        if not NEMO_AVAILABLE:
            raise ImportError(
                "NeMo framework is not installed. Please install NeMo first:\n"
                "pip install nemo_toolkit[all]"
            )
            
        # Get configuration or use defaults
        config = config or {}
            
        # Create STT configuration
        self.config = NVIDIAParakeetConfig(**config)
        
        # Configure logging based on show_logs
        self.configure_logging()
        
        # Setup device
        self._setup_device()
        
        # Load model
        self._load_model()
        
    def configure_logging(self) -> None:
        """Configure logging based on show_logs setting."""
        if not self.config.show_logs:
            logger.setLevel(logging.WARNING)
            # Also set NeMo loggers to WARNING
            for logger_name in ["nemo", "nemo_logger", "nemo.collections.asr"]:
                logging.getLogger(logger_name).setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
    
    def _setup_device(self) -> None:
        """Set up device for model execution."""
        # Setup device
        self.device = f"cuda:{self.config.cuda_device_id}" if torch.cuda.is_available() else "cpu"
        
        # Print device info if logs enabled
        if self.config.show_logs:
            logger.info(f"Device set to use {self.device}")
    
    def _load_model(self) -> None:
        """Load Parakeet-TDT model from checkpoint."""
        # Determine model path
        self.model_path = os.path.join(self.config.model_folder_path, self.config.checkpoint)
        if not os.path.exists(self.model_path) and not self.config.checkpoint.startswith(("http://", "https://")):
            if self.config.show_logs:
                logger.info(f"Using checkpoint name directly: {self.config.checkpoint}")
            self.model_path = self.config.checkpoint
        
        try:
            # Load the model using NeMo's API
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_path
            )
            
            # Move model to the specified device
            self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            if self.config.show_logs:
                logger.info(f"NVIDIA Parakeet-TDT model loaded successfully")
        
        except Exception as e:
            error_msg = f"Error loading Parakeet-TDT model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def update_config(self, **kwargs) -> None:
        """Update STT configuration parameters."""
        # Track changes that require model reloading
        requires_reload = False
        old_show_logs = self.config.show_logs
        
        # Update configuration attributes
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                # Check if this parameter requires model reload
                if key in ["checkpoint", "model_folder_path", "cuda_device_id"]:
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
        
        # Reload model if necessary
        if requires_reload:
            if self.config.show_logs:
                logger.info("Reloading model due to configuration changes")
            self._setup_device()
            self._load_model()
    
    def _process_audio_input(self, audio_data: Any) -> Union[str, np.ndarray]:
        """Process audio data into the format expected by the model."""
        if audio_data is None:
            raise ValueError("No audio data provided")
        
        # Handle file path
        if isinstance(audio_data, str):
            if not os.path.exists(audio_data):
                raise ValueError(f"Audio file not found: {audio_data}")
            return audio_data
        
        # Handle tuple from WebRTC component (sample_rate, audio_array)
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
            
            if isinstance(audio_array, np.ndarray):
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.squeeze()
                
                # Normalize if needed
                if np.abs(audio_array).max() > 1.0:
                    audio_array = audio_array.astype("float32") / 32768.0
                    
                return audio_array
            else:
                raise ValueError(f"Unsupported audio array type: {type(audio_array)}")
        
        # Unsupported type
        raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
    
    def transcribe(self, audio_data: Any, task: str = "transcribe") -> Union[str, Dict[str, Any]]:
        """Transcribe audio data to text using the NVIDIA Parakeet-TDT model."""
        try:
            # Process audio into the right format
            processed_audio = self._process_audio_input(audio_data)
            
            # Transcribe audio using NeMo's API
            if isinstance(processed_audio, str):
                # File path input
                result = self.model.transcribe(
                    audio=[processed_audio], 
                    batch_size=self.config.batch_size,
                    timestamps=True  # Enable timestamp prediction
                )
            else:
                # Direct audio data input
                result = self.model.transcribe(
                    audio=[processed_audio],
                    batch_size=self.config.batch_size,
                    timestamps=True  # Enable timestamp prediction
                )
            
            # Extract transcription from result
            if isinstance(result, list) and len(result) > 0:
                transcription = result[0]
            else:
                transcription = result
            
            # Format result for consistent output
            if isinstance(transcription, str):
                return {"text": transcription}
            elif isinstance(transcription, dict):
                if "text" not in transcription and hasattr(transcription, "text"):
                    transcription["text"] = transcription.text
                return transcription
            else:
                # If result has a text attribute
                if hasattr(transcription, "text"):
                    return {"text": transcription.text}
                # Last resort - convert to string
                return {"text": str(transcription)}
            
        except Exception as e:
            if self.config.show_logs:
                logger.error(f"Error in transcribe: {e}")
            raise ValueError(f"Transcription error: {str(e)}")
            
    def transcribe_stream(self, audio_stream, task: str = "transcribe") -> Iterator[Union[str, Dict[str, Any]]]:
        """Transcribe streaming audio data."""
        if self.config.show_logs:
            logger.info("Using buffered approach for streaming transcription")
        
        buffer: List[np.ndarray] = []
        buffer_duration_seconds = 0
        sample_rate = 16000  # Expected sample rate
        buffer_threshold_seconds = self.config.chunk_length_s
        
        try:
            # Process audio chunks as they come
            for chunk in audio_stream:
                # Process the chunk to get sample rate and numpy array
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    chunk_sample_rate, audio_array = chunk
                    
                    # Skip empty chunks
                    if len(audio_array) == 0:
                        continue
                    
                    # Update sample rate if provided
                    if chunk_sample_rate:
                        sample_rate = chunk_sample_rate
                    
                    # Normalize and process the array
                    if isinstance(audio_array, np.ndarray):
                        if len(audio_array.shape) > 1:
                            audio_array = audio_array.squeeze()
                        
                        # Normalize if needed
                        if np.abs(audio_array).max() > 1.0:
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
                            
                            # Transcribe the combined buffer
                            result = self.model.transcribe(
                                audio=[combined],
                                batch_size=self.config.batch_size,
                                timestamps=True  # Enable timestamp prediction
                            )
                            
                            # Extract transcription
                            if isinstance(result, list) and len(result) > 0:
                                transcription = result[0]
                            else:
                                transcription = result
                            
                            # Format for consistent output
                            if isinstance(transcription, str):
                                yield {"text": transcription}
                            elif isinstance(transcription, dict):
                                if "text" not in transcription and hasattr(transcription, "text"):
                                    transcription["text"] = transcription.text
                                yield transcription
                            else:
                                if hasattr(transcription, "text"):
                                    yield {"text": transcription.text}
                                else:
                                    yield {"text": str(transcription)}
                            
                            # Clear buffer
                            buffer = []
                            buffer_duration_seconds = 0
            
            # Process any remaining audio in buffer
            if buffer:
                combined = np.concatenate(buffer)
                
                # Transcribe the combined buffer
                result = self.model.transcribe(
                    audio=[combined],
                    batch_size=self.config.batch_size,
                    timestamps=True  # Enable timestamp prediction
                )
                
                # Extract transcription
                if isinstance(result, list) and len(result) > 0:
                    transcription = result[0]
                else:
                    transcription = result
                
                # Format for consistent output
                if isinstance(transcription, str):
                    yield {"text": transcription}
                elif isinstance(transcription, dict):
                    if "text" not in transcription and hasattr(transcription, "text"):
                        transcription["text"] = transcription.text
                    yield transcription
                else:
                    if hasattr(transcription, "text"):
                        yield {"text": transcription.text}
                    else:
                        yield {"text": str(transcription)}
                
        except Exception as e:
            if self.config.show_logs:
                logger.error(f"Error in stream transcription: {e}")
            yield {"text": f"Transcription error: {str(e)}", "error": True}