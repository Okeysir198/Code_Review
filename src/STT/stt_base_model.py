## src/STT/stt_base_model.py
"""
Base STT model interface with optimized configuration defaults.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class STTConfig:
    """Base configuration for STT models with optimal defaults."""
    
    # Core settings
    show_logs: bool = False  # Disabled by default for performance
    checkpoint: str = None

    # Audio processing settings (optimized for real-time)
    sampling_rate: int = 16000  # Standard for ASR models
    audio_format: str = "mono"  # Single channel for efficiency
    
    # Performance settings
    use_gpu: bool = True
    cuda_device_id: int = 0
    batch_size: int = None
    
    # Memory optimization
    low_memory_mode: bool = True
    cache_enabled: bool = True
    
    # Additional optimizations
    preprocessing_enabled: bool = True
    normalization_enabled: bool = True


class BaseSTTModel(ABC):
    """Optimized base class for Speech-to-Text models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration validation."""
        self._raw_config = config or {}  # Store raw config
        self._setup_logging()
        
    # In _setup_logging method, change:
    def _setup_logging(self) -> None:
        """Configure optimized logging."""
        show_logs = self._raw_config.get('show_logs', False)  # Get from raw config
        level = logging.INFO if show_logs else logging.WARNING
        logger.setLevel(level)
        
        if not show_logs:
            for framework_logger in ["transformers", "torch", "nemo"]:
                logging.getLogger(framework_logger).setLevel(logging.WARNING)
    
    @abstractmethod
    def transcribe(self, audio_data: Any, **kwargs) -> Union[str, Dict[str, Any]]:
        """Transcribe audio to text with optimal settings."""
        pass
    
    @abstractmethod  
    def transcribe_stream(self, audio_stream, **kwargs) -> Iterator[Union[str, Dict[str, Any]]]:
        """Stream transcription with real-time optimization."""
        pass
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with validation."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                if self.config.show_logs:
                    logger.info(f"Updated config: {key} = {value}")