"""
Configuration settings for the Call Center AI Agent application with client verification.
"""

# Configuration Constants
CONFIG = {
    # General application settings
    "app": {
        "name": "Call Center AI Agent",
        "version": "1.0.0",
        "description": "AI-powered call center agent with client verification",
        "show_logs": True,  # Controls logging output
        "debug_mode": True  # Enables detailed debug information
    },
    "configurable":{
        "use_memory": True,
        "enable_stt_model": True,
        "enable_tts_model": True,

    },
    
    # Ollama chat model config
    "llm": {
        "model_name": "qwen2.5:14b-instruct", #"qwen2.5:7b-instruct-q5_K_M", #phi4-mini:latest, cogito:3b-v1-preview-llama-q4_K_M, qwen2.5:7b-instruct, qwen2.5:3b-instruct
        "temperature": 0.0,
        "max_tokens": 2048,
        "context_window": 8192,  # Maximum tokens in context window
        "timeout": 120,  # Max time in seconds to wait for model response
        "streaming": True,  # Whether to stream responses or wait for complete response
        "trim_factor": 0.75,  # When to trim history if context window is getting full
    },

    # TTS model config with model-specific settings
    "tts": {
        "model_name": "kokorov2",  # Main selector for which TTS model to use
        
        # Kokoro-specific settings
        "kokoro": {
            "accent": "af_heart",  # Voice accent/profile
            "speed": 1.3,
            "language": "en-us",
            "enable_emotions": True,
        },
        
        # Enhanced Kokoro V2 settings
        # https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
        "kokorov2": {
            "voice": "am_michael",  # af_bella, af_heart, am_fenrir, am_michael
            "speed": 1.3,
            "language": "a",  # 'a' for US English, 'b' for UK English
            "use_gpu": True,
            "fallback_to_cpu": True,
            "sample_rate": 24000,
            "preload_voices": ["af_heart"],  # Preload common voices
            "custom_pronunciations": {
                "kokoro": {"a": "kˈOkəɹO", "b": "kˈQkəɹQ"},
                "cartrack": {"a": "kˈɑɹtɹæk", "b": "kˈɑːtɹæk"}
            }
        },
        
    },

    # STT model config with model-specific settings
    "stt": {
        "model_name": "whisper-large-v3-turbo",  # Main selector for which STT model to use
        # "model_name": "nvidia/parakeet-tdt-0.6b-v2",

        # Hugging Face model settings
        "whisper-large-v3-turbo": {
            "checkpoint": "whisper-large-v3-turbo",
            # "model_folder_path": "/media/ct-dev/newSATA_2tb/langgraph/HF_models",
            "model_folder_path": "/home/ct-admin/Documents/Langgraph/HF_models/",
            "batch_size": 8,
            "cuda_device_id": 1,
            "chunk_length_s": 30,
            "compute_type": "float16",  # Computation precision (float16, int8)
            "beam_size": 3  # Beam search size for better transcription accuracy
        },

        # Model-specific configuration
        "nvidia/parakeet-tdt-0.6b-v2": {
            "timestamp_prediction": True,
            "decoding_type": "tdt"  # Use TDT decoder for faster transcription
        }
    },

    # Call script configuration
    "script": {
        "type": "ratio_1_inflow",
    },

    # Name verification config
    "verification": {
        "max_name_verification_attempts": 5,
        "max_details_verification_attempts": 5,
        },

    # Gradio UI config
    "server": {
        "port": 7898,
        "host": "0.0.0.0",
        "share": False,
        "enable_queue": True,
        "max_threads": 40,
        "development_mode": False,
        "enable_analytics": False,
    },


    
}