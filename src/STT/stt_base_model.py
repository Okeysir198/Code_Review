"""
Base STT model interface.

This module defines the abstract base class for all STT implementations,
ensuring a consistent interface regardless of the underlying model.
"""

import abc
from typing import Dict, Any, Optional, Union, Iterator
import numpy as np


class STTConfig:
    """Base configuration class for STT models."""
    
    def __init__(self, **kwargs):
        """Initialize with arbitrary configuration parameters."""
        # Extract common parameters
        self.show_logs = kwargs.get("show_logs", True)
        
        # Store all parameters on the instance
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def __repr__(self) -> str:
        """Return a string representation of the configuration."""
        # Filter out potentially large attributes
        safe_attrs = {k: v for k, v in self.__dict__.items() 
                     if not k.startswith('_') and not isinstance(v, (dict, list)) 
                     or k == 'show_logs'}
        return f"{self.__class__.__name__}({', '.join([f'{k}={v!r}' for k, v in safe_attrs.items()])})"


class BaseSTTModel(abc.ABC):
    """Abstract base class for all STT model implementations.
    
    All STT model implementations must inherit from this class and
    implement its abstract methods to ensure consistent interface.
    """
    
    @abc.abstractmethod
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the STT model with configuration.
        
        Args:
            config: Dictionary with model-specific configuration options
        """
        pass
        
    @abc.abstractmethod
    def update_config(self, **kwargs) -> None:
        """Update model configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        pass
    
    @abc.abstractmethod
    def transcribe(self, audio_data: Any, task: str = "transcribe") -> Union[str, Dict[str, Any]]:
        """Transcribe audio data to text.
        
        Args:
            audio_data: Audio data to transcribe, can be:
                - File path (str)
                - Audio array (numpy.ndarray)
                - Tuple of (sample_rate, audio_array)
            task: Either 'transcribe' (same language) or 'translate' (to English)
            
        Returns:
            Transcribed text as string or dictionary with text and metadata
        """
        pass

    @abc.abstractmethod
    def transcribe_stream(self, audio_stream, task: str = "transcribe") -> Iterator[Union[str, Dict[str, Any]]]:
        """Transcribe streaming audio data.
        
        Args:
            audio_stream: Stream of audio chunks
            task: Either 'transcribe' (same language) or 'translate' (to English)
            
        Yields:
            Transcribed text segments or dictionaries with text and metadata
        """
        pass
    
    def configure_logging(self) -> None:
        """Configure logging based on show_logs configuration.
        
        This method should be implemented by subclasses to handle
        setting appropriate logging levels.
        """
        pass