"""
Optimized Kokoro Text-to-Speech model implementation (V2).

This module provides a wrapper around the Kokoro TTS model,
conforming to the BaseTTSModel interface, with improved configuration
handling and performance optimizations.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Iterator, Optional, Tuple, Union, List
from functools import lru_cache

from kokoro import KModel, KPipeline
import torch
from .tts_base_model import BaseTTSModel, TTSConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KokoroTTSConfigV2(TTSConfig):
    """Configuration for Kokoro TTS model V2.
    
    Attributes:
        voice: Voice accent/profile to use
        speed: Speech rate multiplier (1.0 = normal speed)
        language: Language code identifier
        custom_pronunciations: Custom pronunciation mappings
        use_gpu: Whether to use GPU for inference
        fallback_to_cpu: Whether to fall back to CPU if GPU fails
        sample_rate: Audio sample rate to use
        show_logs: Whether to display log messages
    """
    def __init__(self, **kwargs):
        """Initialize Kokoro TTS V2 configuration with defaults."""
        # Core TTS settings
        self.voice = kwargs.get("voice", "af_heart")
        self.speed = kwargs.get("speed", 1.0)
        self.language = kwargs.get("language", "a")  # 'a' for US English, 'b' for UK English
        
        # Custom pronunciations dictionary
        self.custom_pronunciations = kwargs.get("custom_pronunciations", {
            "kokoro": {"a": "kˈOkəɹO", "b": "kˈQkəɹQ"}
        })
        
        # Performance settings
        self.use_gpu = kwargs.get("use_gpu", torch.cuda.is_available())
        self.fallback_to_cpu = kwargs.get("fallback_to_cpu", True)
        
        # Audio settings
        self.sample_rate = kwargs.get("sample_rate", 24000)
        
        # Voice cache settings
        self.preload_voices = kwargs.get("preload_voices", [self.voice])
        
        # Logging control
        self.show_logs = kwargs.get("show_logs", True)


class KokoroTTSModelV2(BaseTTSModel):
    """Implementation of Kokoro Text-to-Speech model V2 with optimizations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Kokoro TTS model V2 with configuration settings.
        
        Args:
            config: Dictionary with TTS configuration options.
        """
        # Get configuration or use defaults
        config = config or {}
            
        # Create TTS configuration
        self.config = KokoroTTSConfigV2(**config)
        
        # Set logging level based on show_logs configuration
        self._configure_logging()
        
        # Initialize model and pipeline
        self._initialize_model()
        self._initialize_pipelines()
        
        # Preload requested voices
        self._preload_voices()
    
    def _configure_logging(self) -> None:
        """Configure logging based on show_logs setting."""
        if not self.config.show_logs:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
    
    def _initialize_model(self) -> None:
        """Initialize the Kokoro model on the appropriate device."""
        if self.config.show_logs:
            logger.info(f"Initializing Kokoro TTS model V2 with settings: "
                      f"voice={self.config.voice}, speed={self.config.speed}, "
                      f"use_gpu={self.config.use_gpu}")
        
        # Set device based on configuration
        self.device = 'cuda' if self.config.use_gpu and torch.cuda.is_available() else 'cpu'
        
        if self.config.show_logs:
            logger.info(f"Using device: {self.device}")
        
        # Initialize KModel on appropriate device
        self.model = KModel().to(self.device).eval()
    
    def _initialize_pipelines(self) -> None:
        """Initialize language pipelines with custom pronunciations."""
        # Initialize pipelines for different language codes
        self.pipelines = {
            'a': KPipeline(lang_code='a', model=False),  # US English
            'b': KPipeline(lang_code='b', model=False)   # UK English
        }
        
        # Add custom pronunciations from config
        for word, pronunciations in self.config.custom_pronunciations.items():
            for lang_code, pronunciation in pronunciations.items():
                if lang_code in self.pipelines:
                    self.pipelines[lang_code].g2p.lexicon.golds[word] = pronunciation
                    if self.config.show_logs:
                        logger.debug(f"Added custom pronunciation for '{word}' in language '{lang_code}': {pronunciation}")
    
    def _preload_voices(self) -> None:
        """Preload configured voices for faster inference."""
        for voice in self.config.preload_voices:
            self._load_voice(voice)
            if self.config.show_logs:
                logger.debug(f"Preloaded voice: {voice}")
    
    @lru_cache(maxsize=8)
    def _load_voice(self, voice: str) -> Dict:
        """
        Load a voice profile with caching for performance.
        
        Args:
            voice: Voice identifier (e.g., "af_heart")
            
        Returns:
            Voice pack dictionary
        """
        language_code = voice[0]  # First letter of voice ID indicates language
        if language_code not in self.pipelines:
            if self.config.show_logs:
                logger.warning(f"Unsupported language code in voice '{voice}'. Using default 'a'.")
            language_code = 'a'
            
        return self.pipelines[language_code].load_voice(voice)
        
    def update_config(self, **kwargs) -> None:
        """
        Update TTS configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        # Track if GPU setting changed to determine if model needs to be moved
        old_gpu_setting = self.config.use_gpu
        old_show_logs = self.config.show_logs
        
        # Update all provided configuration options
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                if self.config.show_logs:
                    logger.debug(f"Updated config {key} = {value}")
            else:
                if self.config.show_logs:
                    logger.warning(f"Ignoring unknown config parameter: {key}")
        
        # Update logging if show_logs changed
        if old_show_logs != self.config.show_logs:
            self._configure_logging()
        
        # Reinitialize model if GPU setting changed
        if self.config.use_gpu != old_gpu_setting:
            if self.config.show_logs:
                logger.info(f"GPU setting changed from {old_gpu_setting} to {self.config.use_gpu}. Reinitializing model.")
            self._initialize_model()
        
        # Preload any new voices
        if "voice" in kwargs and kwargs["voice"] not in self.config.preload_voices:
            self.config.preload_voices.append(kwargs["voice"])
            self._load_voice(kwargs["voice"])
            
        if self.config.show_logs:
            logger.info(f"Updated TTS configuration: voice={self.config.voice}, speed={self.config.speed}, use_gpu={self.config.use_gpu}")
    
    def _generate_audio(self, text: str, chunk_callback=None) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Generate audio from text with error handling and optional chunking.
        
        Args:
            text: Text to convert to speech
            chunk_callback: Optional callback function for each audio chunk
            
        Yields:
            Tuples containing (sampling_rate, audio_chunk_array)
        """
        if not text or not text.strip():
            if self.config.show_logs:
                logger.warning("Empty text provided for speech synthesis")
            yield self.config.sample_rate, np.zeros(1, dtype=np.float32)
            return
            
        # Get language code from voice
        language_code = self.config.voice[0]
        if language_code not in self.pipelines:
            if self.config.show_logs:
                logger.warning(f"Unsupported language code in voice '{self.config.voice}'. Using default 'a'.")
            language_code = 'a'
            
        pipeline = self.pipelines[language_code]
        
        # Load voice pack
        pack = self._load_voice(self.config.voice)
        
        # Process text using pipeline
        first_chunk = True
        for _, ps, _ in pipeline(text, self.config.voice, self.config.speed):
            ref_s = pack[len(ps)-1]
            
            # Generate audio with error handling
            try:
                # Try on configured device first
                audio = self.model(ps, ref_s, self.config.speed)
                
            except Exception as e:
                if self.config.show_logs:
                    logger.error(f"Error during TTS generation: {str(e)}")
                # Fallback to CPU if configured and initial attempt failed
                if self.config.use_gpu and self.config.fallback_to_cpu:
                    if self.config.show_logs:
                        logger.info("Falling back to CPU for TTS generation")
                    cpu_model = self.model.to('cpu')
                    audio = cpu_model(ps, ref_s, self.config.speed)
                else:
                    if self.config.show_logs:
                        logger.error("TTS generation failed and fallback to CPU disabled")
                    raise
            
            # Convert to numpy array
            audio_np = audio.cpu().numpy()
            
            # Apply any callback processing
            if chunk_callback:
                audio_np = chunk_callback(audio_np, first_chunk)
            
            # Yield the audio chunk
            yield self.config.sample_rate, audio_np
            
            # Add a silent chunk after the first audio segment if needed
            if first_chunk:
                first_chunk = False
                yield self.config.sample_rate, np.zeros(1, dtype=np.float32)
    
    def text_to_speech(self, text: str) -> Tuple[int, np.ndarray]:
        """
        Convert text to speech and return full audio data with sampling rate.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Tuple containing (sampling_rate, audio_array)
        """
        start_time = time.time()
        if self.config.show_logs:
            logger.debug(f"Converting to speech: {text[:50]}...")
        
        # Generate first chunk only
        for sample_rate, audio_np in self._generate_audio(text):
            if self.config.show_logs:
                logger.debug(f"TTS conversion completed in {time.time() - start_time:.3f}s")
            return sample_rate, audio_np
            
        # Fallback empty audio if no chunks generated
        if self.config.show_logs:
            logger.warning("No audio was generated")
        return self.config.sample_rate, np.zeros(1, dtype=np.float32)
        
    def stream_text_to_speech(self, text: str) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Stream text to speech in chunks for real-time playback.
        
        Args:
            text: Text to convert to speech
            
        Yields:
            Tuples containing (sampling_rate, audio_chunk_array)
        """
        start_time = time.time()
        if self.config.show_logs:
            logger.debug(f"Streaming speech: {text[:50]}...")
        
        # Generate and yield all audio chunks
        yield from self._generate_audio(text)
        
        if self.config.show_logs:
            logger.debug(f"TTS streaming completed in {time.time() - start_time:.3f}s")