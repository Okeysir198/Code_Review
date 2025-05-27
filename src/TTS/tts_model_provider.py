"""
TTS model provider module.

This module provides functionality to create and manage TTS model instances
based on configuration, abstracting the specific implementation details.
"""

import logging
from typing import Dict, Any, Optional, Type

from .tts_base_model import BaseTTSModel
from .tts_kokoro_model import KokoroTTSModel
from .tts_kokoro_modelv2 import KokoroTTSModelV2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Registry of available TTS models
TTS_MODELS = {
    "kokoro": KokoroTTSModel,
    "kokorov2": KokoroTTSModelV2,
}


def create_tts_model(config: Dict[str, Any]) -> BaseTTSModel:
    """
    Create and return a TTS model based on configuration.
    
    Args:
        config: Dictionary containing TTS configuration with a structure like:
               {
                   "model_name": "kokoro",
                   "show_logs": True,
                   "kokoro": { ... kokoro-specific settings ... },
               }
               
    Returns:
        Initialized TTS model instance
        
    Raises:
        ValueError: If model_name is not provided or not supported
    """
    # Extract model name and logging preference from config
    model_name = config.get("model_name", "kokoro")
    show_logs = config.get("show_logs", True)
    
    # Check if model is supported
    if model_name not in TTS_MODELS:
        supported_models = ", ".join(TTS_MODELS.keys())
        if show_logs:
            logger.warning(f"Unsupported TTS model: {model_name}. Falling back to 'kokoro'. "
                          f"Supported models: {supported_models}")
        model_name = "kokoro"
    
    # Get model-specific settings
    if model_name not in config:
        if show_logs:
            logger.warning(f"No configuration found for {model_name}. Using default settings.")
        model_config = {}
    else:
        model_config = config[model_name].copy()  # Create a copy to avoid modifying the original
    
    # Add show_logs parameter to model config
    model_config["show_logs"] = show_logs
    
    # Log configuration if logs are enabled
    if show_logs:
        # Only log essential config information to avoid exposing sensitive data
        log_config = {k: v for k, v in model_config.items() if k not in ("custom_pronunciations")}
        logger.info(f"Creating {model_name} TTS model with config: {log_config}")
    
    # Get model class and create instance
    model_class = TTS_MODELS[model_name]
    return model_class(model_config)


def register_tts_model(name: str, model_class: Type[BaseTTSModel]) -> None:
    """
    Register a new TTS model implementation.
    
    Args:
        name: Name to register the model under
        model_class: TTS model class (must inherit from BaseTTSModel)
        
    Raises:
        TypeError: If model_class does not inherit from BaseTTSModel
    """
    # Check if model class inherits from BaseTTSModel
    if not issubclass(model_class, BaseTTSModel):
        raise TypeError(f"Model class must inherit from BaseTTSModel, got {model_class.__name__}")
    
    # Register model
    TTS_MODELS[name] = model_class
    logger.info(f"Registered TTS model: {name}")


def get_available_models() -> Dict[str, Type[BaseTTSModel]]:
    """
    Get dictionary of available TTS models.
    
    Returns:
        Dictionary mapping model names to model classes
    """
    return TTS_MODELS.copy()


def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific TTS model.
    
    Args:
        model_name: Name of the model to get information for
        
    Returns:
        Dictionary with model information or None if model not found
    """
    if model_name not in TTS_MODELS:
        return None
        
    model_class = TTS_MODELS[model_name]
    
    # Get model info from docstring and class attributes
    return {
        "name": model_name,
        "class": model_class.__name__,
        "description": model_class.__doc__.strip().split('\n')[0] if model_class.__doc__ else "No description available",
        "capabilities": getattr(model_class, "CAPABILITIES", []),
    }