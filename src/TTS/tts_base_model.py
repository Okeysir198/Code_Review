"""
Base TTS model interface.

This module defines the abstract base class for all TTS implementations,
ensuring a consistent interface regardless of the underlying model.
"""

import abc
from typing import Dict, Any, Iterator, Optional


class TTSConfig:
    """Base configuration class for TTS models."""
    
    def __init__(self, **kwargs):
        """Initialize with arbitrary configuration parameters."""
        for key, value in kwargs.items():
            setattr(self, key, value)


class BaseTTSModel(abc.ABC):
    """Abstract base class for all TTS model implementations.
    
    All TTS model implementations must inherit from this class and
    implement its abstract methods to ensure consistent interface.
    """
    
    @abc.abstractmethod
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the TTS model with configuration.
        
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
    def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech and return full audio data.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Complete audio data as bytes
        """
        pass
        
    @abc.abstractmethod
    def stream_text_to_speech(self, text: str) -> Iterator[bytes]:
        """Stream text to speech in chunks for real-time playback.
        
        Args:
            text: Text to convert to speech
            
        Yields:
            Audio data chunks as bytes
        """
        pass