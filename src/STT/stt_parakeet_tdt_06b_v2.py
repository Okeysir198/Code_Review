## src/STT/stt_parakeet_tdt_06b_v2.py
"""
Optimized NVIDIA Parakeet-TDT model - Simple and lean version.
"""

import os
import torch
import numpy as np
import logging
import tempfile
import soundfile as sf
from typing import Dict, Any, Optional, Union, Iterator
from dataclasses import dataclass

try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

from .stt_base_model import BaseSTTModel, STTConfig

logger = logging.getLogger(__name__)

@dataclass
class NVIDIAParakeetConfig(STTConfig):
    """Optimized configuration for NVIDIA Parakeet-TDT models."""
    
    # Model settings - optimized for parakeet-tdt-0.6b-v2
    checkpoint: str = "nvidia/parakeet-tdt-0.6b-v2"
    model_folder_path: str = "./models"
    
    # Performance optimizations (based on HF leaderboard settings)
    batch_size: int = 32  # Optimal for RTFx performance
    chunk_length_s: int = 24 * 60  # Max 24 minutes per chunk
    
    # Audio processing optimizations
    sampling_rate: int = 16000  # Native sampling rate for Parakeet
    
    # TDT-specific optimizations
    timestamp_prediction: bool = True  # Leverages TDT decoder
    decoding_type: str = "tdt"  # Use TDT decoder for speed
    
    # Memory optimizations
    compute_timestamps: bool = True  # Word-level timestamps


class NVIDIAParakeetModel(BaseSTTModel):
    """Simple NVIDIA Parakeet-TDT implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if not NEMO_AVAILABLE:
            raise ImportError("NeMo framework required. Install with: pip install nemo_toolkit[all]")
            
        super().__init__(config)
        self.config = NVIDIAParakeetConfig(**(self._raw_config or {}))
        self._setup_device()
        self._load_model()
        
    def _setup_device(self) -> None:
        self.device = f"cuda:{self.config.cuda_device_id}" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
        if self.config.show_logs:
            logger.info(f"Device: {self.device}")
    
    def _load_model(self) -> None:
        model_path = os.path.join(self.config.model_folder_path, self.config.checkpoint)
        if not os.path.exists(model_path):
            model_path = self.config.checkpoint
        
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.config.show_logs:
            logger.info(f"Loaded Parakeet-TDT model successfully")
    
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
    
    def transcribe(self, audio_data: Any, **kwargs) -> Union[str, Dict[str, Any]]:
        """Transcribe with working Parakeet settings and return timestamps."""
        try:
            processed_audio = self._process_audio_input(audio_data)
            
            transcription_kwargs = {
                'batch_size': self.config.batch_size,
                'timestamps': self.config.timestamp_prediction,
                'return_hypotheses': False,
            }
            
            # Handle different input types properly
            if isinstance(processed_audio, str):
                # File path - direct transcription
                result = self.model.transcribe(audio=[processed_audio], **transcription_kwargs)
            else:
                # Numpy array - need temp file for NeMo
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    sf.write(tmp_file.name, processed_audio, self.config.sampling_rate)
                    result = self.model.transcribe(audio=[tmp_file.name], **transcription_kwargs)
                    os.unlink(tmp_file.name)
            
            # Format result consistently with timestamps
            if isinstance(result, list) and len(result) > 0:
                transcription = result[0]
            else:
                transcription = result
            
            # Extract text and timestamps
            output = {}
            
            if isinstance(transcription, str):
                output["text"] = transcription
            elif isinstance(transcription, dict):
                output["text"] = transcription.get("text", "")
                # Include timestamps if available
                if "timestamp" in transcription:
                    output["timestamps"] = transcription["timestamp"]
            elif hasattr(transcription, "text"):
                output["text"] = transcription.text
                # Check for timestamps attribute
                if hasattr(transcription, "timestamp"):
                    output["timestamps"] = transcription.timestamp
            else:
                output["text"] = str(transcription)
            
            return output
                
        except Exception as e:
            if self.config.show_logs:
                logger.error(f"Transcription error: {e}")
            raise ValueError(f"Transcription error: {str(e)}")

    def transcribe_stream(self, audio_stream, **kwargs) -> Iterator[Union[str, Dict[str, Any]]]:
        """Enhanced streaming with proper buffering and timestamps."""
        audio_buffer = []
        buffer_duration = 0
        target_duration = 10  # seconds
        total_processed_duration = 0  # Track total time for timestamp adjustment
        
        for chunk in audio_stream:
            try:
                # Handle different chunk formats
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    sample_rate, audio_array = chunk
                    
                    if len(audio_array) == 0:
                        continue
                    
                    # Normalize audio properly
                    if isinstance(audio_array, np.ndarray):
                        audio_array = audio_array.squeeze()
                        
                        # Convert to float32 and normalize
                        if audio_array.dtype != np.float32:
                            audio_array = audio_array.astype(np.float32)
                            if np.abs(audio_array).max() > 1.0:
                                audio_array = audio_array / 32768.0
                    else:
                        continue  # Skip invalid chunks
                    
                    audio_buffer.append(audio_array)
                    chunk_duration = len(audio_array) / self.config.sampling_rate
                    buffer_duration += chunk_duration
                    
                    # Process when buffer reaches target duration
                    if buffer_duration >= target_duration:
                        combined_audio = np.concatenate(audio_buffer)
                        
                        # Transcribe with timestamps
                        result = self.transcribe(combined_audio)
                        
                        # Adjust timestamps for streaming context
                        if result and "timestamps" in result:
                            # Adjust timestamps by adding total processed duration
                            adjusted_timestamps = []
                            for timestamp_info in result["timestamps"]:
                                if isinstance(timestamp_info, dict):
                                    adjusted_info = timestamp_info.copy()
                                    if "start" in adjusted_info:
                                        adjusted_info["start"] += total_processed_duration
                                    if "end" in adjusted_info:
                                        adjusted_info["end"] += total_processed_duration
                                    adjusted_timestamps.append(adjusted_info)
                                else:
                                    adjusted_timestamps.append(timestamp_info)
                            result["timestamps"] = adjusted_timestamps
                        
                        # Add stream metadata
                        if result:
                            result["stream_chunk_start"] = total_processed_duration
                            result["stream_chunk_end"] = total_processed_duration + buffer_duration
                            yield result
                        
                        # Update total processed time
                        total_processed_duration += buffer_duration
                        
                        # Reset buffer
                        audio_buffer = []
                        buffer_duration = 0
                
                elif isinstance(chunk, np.ndarray):
                    # Direct numpy array input
                    if len(chunk) == 0:
                        continue
                    
                    audio_array = chunk.squeeze().astype(np.float32)
                    if np.abs(audio_array).max() > 1.0:
                        audio_array = audio_array / 32768.0
                    
                    audio_buffer.append(audio_array)
                    buffer_duration += len(audio_array) / self.config.sampling_rate
                    
                    if buffer_duration >= target_duration:
                        combined_audio = np.concatenate(audio_buffer)
                        result = self.transcribe(combined_audio)
                        
                        if result:
                            result["stream_chunk_start"] = total_processed_duration
                            result["stream_chunk_end"] = total_processed_duration + buffer_duration
                            yield result
                        
                        total_processed_duration += buffer_duration
                        audio_buffer = []
                        buffer_duration = 0
                            
            except Exception as e:
                if self.config.show_logs:
                    logger.warning(f"Stream chunk error: {e}")
                continue
        
        # Process any remaining buffer
        if audio_buffer and buffer_duration > 1.0:  # Only if meaningful duration
            try:
                combined_audio = np.concatenate(audio_buffer)
                result = self.transcribe(combined_audio)
                
                # Adjust final timestamps
                if result and "timestamps" in result:
                    adjusted_timestamps = []
                    for timestamp_info in result["timestamps"]:
                        if isinstance(timestamp_info, dict):
                            adjusted_info = timestamp_info.copy()
                            if "start" in adjusted_info:
                                adjusted_info["start"] += total_processed_duration
                            if "end" in adjusted_info:
                                adjusted_info["end"] += total_processed_duration
                            adjusted_timestamps.append(adjusted_info)
                        else:
                            adjusted_timestamps.append(timestamp_info)
                    result["timestamps"] = adjusted_timestamps
                
                if result:
                    result["stream_chunk_start"] = total_processed_duration
                    result["stream_chunk_end"] = total_processed_duration + buffer_duration
                    result["is_final_chunk"] = True
                    yield result
                    
            except Exception as e:
                if self.config.show_logs:
                    logger.warning(f"Final buffer error: {e}")