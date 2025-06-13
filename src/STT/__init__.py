## src/STT/__init__.py
"""
Optimized STT package for Call Center AI Agent.
High-performance Speech-to-Text functionality with optimal configurations.
"""

from .stt_base_model import BaseSTTModel, STTConfig
from .stt_whisper_large_v3_turbo import OPENAIWhisperV3TurboModel, OPENAIWhisperV3TurboConfig  
from .stt_parakeet_tdt_06b_v2 import NVIDIAParakeetModel, NVIDIAParakeetConfig
from .stt_model_provider import create_stt_model, get_optimal_config

__all__ = [
    "BaseSTTModel", 
    "STTConfig",
    "OPENAIWhisperV3TurboModel", 
    "OPENAIWhisperV3TurboConfig",
    "NVIDIAParakeetModel",
    "NVIDIAParakeetConfig", 
    "create_stt_model",
    "get_optimal_config"
]