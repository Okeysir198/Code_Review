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

    "audio": {
        "noise_suppression": {
            "enabled": True,
            "method": "webrtc_enhanced",  # webrtc_enhanced,
            "aggressiveness": "medium",   # low, medium, high
            "vad_threshold": 50.0,        # Voice Activity Detection threshold
            "grace_period_ms": 200        # Grace period for voice detection
        },
        "echo_cancellation": {
            "enabled": True,
            "aggressive": True
        },
        "auto_gain_control": {
            "enabled": True,
            "target_level": -18           # dB target level
        }
    },

    "turn_detection": {
        "enabled": True,
        "model_name": "multilingual",  # or "eou" for End-of-Utterance
        "confidence_threshold": 0.7,
        "min_speech_duration": 0.5,  # Minimum speech duration before considering turn
        "max_silence_duration": 2.0, # Maximum silence before forcing turn
        "fallback_to_vad": True, 
        "context_window_turns": 4 # Number of previous turns to consider
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
        # Use Parakeet-TDT as default (currently top ASR leaderboard)
        "model_name": "nvidia/parakeet-tdt-0.6b-v2",
        "show_logs": False,  # Performance optimization
        
        # Whisper-large-v3-turbo optimized settings
        "openai/whisper-large-v3-turbo": {
            "checkpoint": "openai/whisper-large-v3-turbo",
            # "model_folder_path": "",
            "batch_size": 4,  # Optimal balance
            "cuda_device_id": 1,
            "chunk_length_s": 30,  # Whisper native
            "compute_type": "float16",
            "beam_size": 1,  # Greedy for speed
            "condition_on_prev_tokens": False,  # Reduce hallucinations
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "return_timestamps": True,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        },
        
        # NVIDIA Parakeet-TDT optimized settings
        "nvidia/parakeet-tdt-0.6b-v2": {
            "checkpoint": "nvidia/parakeet-tdt-0.6b-v2",
            # "model_folder_path": "",
            "batch_size": 32,  # Optimal for RTFx 3380
            "cuda_device_id": 1,
            "chunk_length_s": 24 * 60,  # 24 minutes max
            "sampling_rate": 16000,  # Native rate
            "timestamp_prediction": True,
            "decoding_type": "tdt",  # Use TDT decoder
            "compute_timestamps": True,
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