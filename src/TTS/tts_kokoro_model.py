"""
Kokoro Text-to-Speech model implementation.

This module provides a wrapper around the Kokoro TTS model,
conforming to the BaseTTSModel interface.
"""

import time
import logging
from typing import Dict, Any, Iterator, Optional

from fastrtc import get_tts_model, KokoroTTSOptions
from .tts_base_model import BaseTTSModel, TTSConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KokoroTTSConfig(TTSConfig):
    """Configuration for Kokoro TTS model.
    
    Attributes:
        voice: Voice accent/profile to use
        speed: Speech rate multiplier (1.0 = normal speed)
        language: Language code (e.g., 'en-us')
    """
    def __init__(self, **kwargs):
        """Initialize Kokoro TTS configuration with defaults."""
        self.voice = kwargs.get("voice", "af_heart")
        self.speed = kwargs.get("speed", 1.0)
        self.language = kwargs.get("language", "en-us")


class KokoroTTSModel(BaseTTSModel):
    """Implementation of Kokoro Text-to-Speech model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Kokoro TTS model with configuration settings.
        
        Args:
            config: Dictionary with TTS configuration options.
        """
        # Get configuration or use defaults
        if config is None:
            config = {}
            
        # Create TTS configuration
        self.config = KokoroTTSConfig(
            voice=config.get("accent", "af_heart"),
            speed=config.get("speed", 1.0),
            language=config.get("language", "en-us")
        )
        
        # Initialize TTS options
        self.tts_options = KokoroTTSOptions(
            voice=self.config.voice,
            speed=self.config.speed,
            lang=self.config.language
        )
        
        # Load TTS model
        logger.info(f"Initializing Kokoro TTS model with voice: {self.config.voice}, "
                    f"speed: {self.config.speed}, language: {self.config.language}")
        self.model = get_tts_model(model="kokoro")
        
    def update_config(self, **kwargs) -> None:
        """
        Update TTS configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
                - voice: Voice accent/profile
                - speed: Speech rate multiplier
                - language: Language code
        """
        # Update configuration
        if "voice" in kwargs:
            self.config.voice = kwargs["voice"]
            self.tts_options.voice = kwargs["voice"]
            
        if "speed" in kwargs:
            self.config.speed = kwargs["speed"]
            self.tts_options.speed = kwargs["speed"]
            
        if "language" in kwargs:
            self.config.language = kwargs["language"]
            self.tts_options.lang = kwargs["language"]
            
        logger.info(f"Updated TTS configuration: {self.config.__dict__}")
    
    def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech and return full audio data.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Complete audio data as bytes
        """
        start_time = time.time()
        logger.debug(f"Converting to speech: {text[:50]}...")
        
        # Get full audio
        audio_data = self.model.tts_sync(text, options=self.tts_options)
        
        logger.debug(f"TTS conversion completed in {time.time() - start_time:.3f}s")
        return audio_data
        
    def stream_text_to_speech(self, text: str) -> Iterator[bytes]:
        """
        Stream text to speech in chunks for real-time playback.
        
        Args:
            text: Text to convert to speech
            
        Yields:
            Audio data chunks as bytes
        """
        start_time = time.time()
        logger.debug(f"Streaming speech: {text[:50]}...")
        
        # Stream audio chunks
        for chunk in self.model.stream_tts_sync(text, options=self.tts_options):
            yield chunk
            
        logger.debug(f"TTS streaming completed in {time.time() - start_time:.3f}s")