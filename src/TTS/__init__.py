"""
TTS package for Call Center AI Agent.

This package provides Text-to-Speech functionality for the Call Center AI Agent.
"""

from .tts_base_model import BaseTTSModel, TTSConfig
from .tts_kokoro_model import KokoroTTSModel, KokoroTTSConfig
from .tts_kokoro_modelv2 import KokoroTTSModelV2, KokoroTTSConfigV2
from .tts_model_provider import create_tts_model, register_tts_model

__all__ = [
    "BaseTTSModel", 
    "TTSConfig",
    "KokoroTTSModel", 
    "KokoroTTSConfig",
    "KokoroTTSModelV2", 
    "KokoroTTSConfigV2",
    "create_tts_model",
    "register_tts_model"
]