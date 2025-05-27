"""
STT model provider module.

This module provides functionality to create and manage STT model instances
based on configuration, abstracting the specific implementation details.
"""

import logging
from typing import Dict, Any, Type, Optional

from .stt_base_model import BaseSTTModel
from .stt_hf_model import HFSTTModel
from .stt_nvidia_model import NVIDIAParakeetModel


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Registry of available STT models
STT_MODELS = {
    "whisper-large-v3-turbo": HFSTTModel,
    "nvidia/parakeet-tdt-0.6b-v2": NVIDIAParakeetModel,

}


def create_stt_model(config: Dict[str, Any]) -> BaseSTTModel:
    """
    Create and return an STT model based on configuration.
    
    Args:
        config: Dictionary containing STT configuration with a structure like:
               {
                   "model_name": "whisper-large-v3-turbo",
                   "show_logs": True,
                   "whisper-large-v3-turbo": { ... model-specific settings ... },
               }
               
    Returns:
        Initialized STT model instance
        
    Raises:
        ValueError: If model_name is not supported
    """
    # Extract model name and logging preference from config
    model_name = config.get("model_name", "whisper-large-v3-turbo")
    show_logs = config.get("show_logs", True)
    
    # Check if model is supported
    if model_name not in STT_MODELS:
        supported_models = ", ".join(STT_MODELS.keys())
        if show_logs:
            logger.warning(f"Unsupported STT model: {model_name}. Falling back to 'whisper-large-v3-turbo'. "
                          f"Supported models: {supported_models}")
        model_name = "whisper-large-v3-turbo"
    
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
        # Log only non-sensitive configuration parameters
        safe_config = {k: v for k, v in model_config.items() 
                      if not isinstance(v, dict) and not k.startswith('_')}
        logger.info(f"Creating {model_name} STT model with config: {safe_config}")
    
    # Get model class and create instance
    model_class = STT_MODELS[model_name]
    return model_class(model_config)


def register_stt_model(name: str, model_class: Type[BaseSTTModel]) -> None:
    """
    Register a new STT model implementation.
    
    Args:
        name: Name to register the model under
        model_class: STT model class (must inherit from BaseSTTModel)
        
    Raises:
        TypeError: If model_class does not inherit from BaseSTTModel
    """
    # Check if model class inherits from BaseSTTModel
    if not issubclass(model_class, BaseSTTModel):
        raise TypeError(f"Model class must inherit from BaseSTTModel, got {model_class.__name__}")
    
    # Register model
    STT_MODELS[name] = model_class
    logger.info(f"Registered STT model: {name}")


def get_available_models() -> Dict[str, Type[BaseSTTModel]]:
    """
    Get dictionary of available STT models.
    
    Returns:
        Dictionary mapping model names to model classes
    """
    return STT_MODELS.copy()


def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific STT model.
    
    Args:
        model_name: Name of the model to get information for
        
    Returns:
        Dictionary with model information or None if model not found
    """
    if model_name not in STT_MODELS:
        return None
        
    model_class = STT_MODELS[model_name]
    
    # Get model info from docstring and class attributes
    return {
        "name": model_name,
        "class": model_class.__name__,
        "description": model_class.__doc__.strip().split('\n')[0] if model_class.__doc__ else "No description available",
        "capabilities": getattr(model_class, "CAPABILITIES", []),
    }