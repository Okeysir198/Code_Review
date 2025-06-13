## src/STT/stt_model_provider.py
"""
Optimized STT model provider with best-practice configurations.
"""

import logging
from typing import Dict, Any, Type, Optional

from .stt_base_model import BaseSTTModel
from .stt_whisper_large_v3_turbo import OPENAIWhisperV3TurboModel, OPENAIWhisperV3TurboConfig
from .stt_parakeet_tdt_06b_v2 import NVIDIAParakeetModel, NVIDIAParakeetConfig

logger = logging.getLogger(__name__)

# Registry with optimized models
STT_MODELS = {
    "openai/whisper-large-v3-turbo": OPENAIWhisperV3TurboModel,
    "nvidia/parakeet-tdt-0.6b-v2": NVIDIAParakeetModel,
}

# Optimal configurations based on benchmarks and best practices
OPTIMAL_CONFIGS = {
    "whisper-large-v3-turbo": {
        "checkpoint": "openai/whisper-large-v3-turbo",
        "batch_size": 4,  # Optimal for RTF
        "chunk_length_s": 30,  # Native Whisper chunk size
        "compute_type": "float16",
        "beam_size": 3,  # Greedy decoding for speed
        "condition_on_prev_tokens": False,  # Reduces hallucinations
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "return_timestamps": True,
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
    },
    
    "nvidia/parakeet-tdt-0.6b-v2": {
        "checkpoint": "nvidia/parakeet-tdt-0.6b-v2", 
        "batch_size": 32,  # Optimal for RTFx 3380 performance
        "chunk_length_s": 24 * 60,  # Max 24 minutes
        "sampling_rate": 16000,  # Native sampling rate
        "timestamp_prediction": True,
        "decoding_type": "tdt",  # Use TDT decoder
        "compute_timestamps": True,
    }
}


def get_optimal_config(model_name: str) -> Dict[str, Any]:
    """Get optimal configuration for specified model."""
    if model_name in OPTIMAL_CONFIGS:
        return OPTIMAL_CONFIGS[model_name].copy()
    
    logger.warning(f"No optimal config for {model_name}, using defaults")
    return {}


def create_stt_model(config: Dict[str, Any]) -> BaseSTTModel:
    """Create optimized STT model with best-practice settings."""
    
    model_name = config.get("model_name", "nvidia/parakeet-tdt-0.6b-v2")  # Default to Parakeet
    show_logs = config.get("show_logs", False)  # Default to quiet
    
    # Validate model support
    if model_name not in STT_MODELS:
        supported_models = ", ".join(STT_MODELS.keys())
        logger.warning(f"Unsupported model: {model_name}. Defaulting to nvidia/parakeet-tdt-0.6b-v2")
        logger.info(f"Supported models: {supported_models}")
        model_name = "nvidia/parakeet-tdt-0.6b-v2"
    
    # Get optimal configuration
    optimal_config = get_optimal_config(model_name)
    
    # Merge user config with optimal defaults
    model_config = optimal_config.copy()
    if model_name in config:
        model_config.update(config[model_name])
    
    # Add global settings
    model_config["show_logs"] = show_logs
    
    # Log configuration if enabled
    if show_logs:
        safe_config = {k: v for k, v in model_config.items() 
                      if not isinstance(v, dict) and not k.startswith('_')}
        logger.info(f"Creating {model_name} with optimal config: {safe_config}")
    
    # Create model instance
    model_class = STT_MODELS[model_name]
    return model_class(model_config)


def register_stt_model(name: str, model_class: Type[BaseSTTModel], 
                      optimal_config: Optional[Dict[str, Any]] = None) -> None:
    """Register new STT model with optional optimal configuration."""
    STT_MODELS[name] = model_class
    
    if optimal_config:
        OPTIMAL_CONFIGS[name] = optimal_config
    
    logger.info(f"Registered STT model: {name}")
