"""
LiveKit Turn Detection Implementation
Optimized version with clean structure and proper error handling
"""

import json
import time
from typing import List, Dict, Any, Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download


class LiveKitTurnDetector:
    """
    Optimized replication of LiveKit's turn detection implementation
    Using the correct model versions and logic with improved structure
    """
    
    # Class constants
    HG_MODEL = "livekit/turn-detector"
    MODEL_REVISIONS = {
        "en": "v1.2.2-en",
        "multilingual": "v0.2.0-intl",
    }
    ONNX_FILENAME = "model_q8.onnx"
    MAX_HISTORY_TOKENS = 128
    MAX_HISTORY_TURNS = 2
    
    # Language-specific thresholds for turn detection
    LANGUAGE_THRESHOLDS = {
        "en": 0.5, "es": 0.5, "fr": 0.5, "de": 0.5, "zh": 0.5,
        "ja": 0.5, "ko": 0.5, "pt": 0.5, "it": 0.5, "nl": 0.5,
        "ru": 0.5, "tr": 0.5, "id": 0.5
    }
    
    def __init__(self, model_type: str = "multilingual"):
        """
        Initialize with the correct LiveKit model version
        
        Args:
            model_type: Either "multilingual" or "en"
        """
        self.session: Optional[ort.InferenceSession] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model_revision: Optional[str] = None
        self.initialize(model_type)

        
    def initialize(self, model_type: str = "multilingual") -> bool:
        """
        Initialize with the correct LiveKit model version
        
        Args:
            model_type: Either "multilingual" or "en"
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if model_type not in self.MODEL_REVISIONS:
            print(f"✗ Invalid model type: {model_type}. Choose from: {list(self.MODEL_REVISIONS.keys())}")
            return False
            
        print(f"Initializing LiveKit turn detector ({model_type})...")
        self.model_revision = self.MODEL_REVISIONS[model_type]
        print(f"Using model revision: {self.model_revision}")
        
        try:
            self._load_onnx_model()
            self._load_tokenizer()
            return True
            
        except Exception as e:
            print(f"✗ Initialization failed: {e}")
            return False
    
    def _load_onnx_model(self) -> None:
        """Load the ONNX model from HuggingFace Hub"""
        local_path_onnx = hf_hub_download(
            self.HG_MODEL,
            self.ONNX_FILENAME,
            subfolder="onnx",
            revision=self.model_revision,
            local_files_only=False
        )
        
        self.session = ort.InferenceSession(
            local_path_onnx,
            providers=["CPUExecutionProvider"]
        )
        print(f"✓ ONNX model loaded: {local_path_onnx}")
    
    def _load_tokenizer(self) -> None:
        """Load the tokenizer from HuggingFace Hub"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.HG_MODEL,
            revision=self.model_revision,
            truncation_side="left"
        )
        print("✓ Tokenizer loaded")
    
    def _validate_initialization(self) -> None:
        """Validate that the model is properly initialized"""
        if not self.session or not self.tokenizer:
            raise RuntimeError("Model not initialized. Call initialize() first.")
    
    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """Validate input messages format"""
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
    
    def _format_chat_context(self, chat_ctx: List[Dict[str, str]]) -> str:
        """
        Format chat context exactly like LiveKit's implementation
        
        Args:
            chat_ctx: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            str: Formatted conversation text
        """
        # Filter out empty messages
        filtered_msgs = [
            {**msg, "content": msg.get("content", "")}
            for msg in chat_ctx
            if msg.get("content", "").strip()
        ]
        
        if not filtered_msgs:
            return ""
        
        # Apply chat template
        convo_text = self.tokenizer.apply_chat_template(
            filtered_msgs,
            add_generation_prompt=False,
            add_special_tokens=False,
            tokenize=False,
        )
        
        # Remove the EOU token from current utterance
        end_token_idx = convo_text.rfind("<|im_end|>")
        return convo_text[:end_token_idx] if end_token_idx != -1 else convo_text
    
    def _run_inference(self, text: str) -> np.ndarray:
        """
        Run ONNX inference on the formatted text
        
        Args:
            text: Formatted conversation text
            
        Returns:
            np.ndarray: Model output
        """
        inputs = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="np",
            max_length=self.MAX_HISTORY_TOKENS,
            truncation=True,
        )
        
        outputs = self.session.run(
            None, 
            {"input_ids": inputs["input_ids"].astype("int64")}
        )
        
        return outputs[0]
    
    def predict_eou(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Predict end-of-utterance probability
        
        Args:
            messages: List of conversation messages
            
        Returns:
            dict: Prediction results including EOU probability
            
        Raises:
            RuntimeError: If model not initialized
            ValueError: If messages are invalid
        """
        start_time = time.perf_counter()
        
        try:
            self._validate_initialization()
            self._validate_messages(messages)
            
            # Format and validate text
            text = self._format_chat_context(messages)
            if not text:
                raise ValueError("No valid content found in messages")
            
            # Run inference
            outputs = self._run_inference(text)
            eou_probability = outputs.flatten()[-1]
            
            duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
            
            return {
                "eou_probability": float(eou_probability),
                "input": text,
                "duration_ms": duration_ms,
                "raw_output_shape": outputs.shape
            }
            
        except Exception as e:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
            return {
                "eou_probability": 0.0,
                "input": "",
                "duration_ms": duration_ms,
                "error": str(e)
            }
    
    def get_language_threshold(self, language: str = "en") -> float:
        """
        Get the recommended threshold for a specific language
        
        Args:
            language: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            float: Recommended threshold value
        """
        lang = language.lower()
        
        # Try exact match first
        if lang in self.LANGUAGE_THRESHOLDS:
            return self.LANGUAGE_THRESHOLDS[lang]
        
        # Try base language if full code not found (e.g., 'en-US' -> 'en')
        if "-" in lang:
            base_lang = lang.split("-")[0]
            if base_lang in self.LANGUAGE_THRESHOLDS:
                return self.LANGUAGE_THRESHOLDS[base_lang]
        
        # Default threshold
        return 0.5
    
    def _get_confidence_level(self, eou_prob: float) -> str:
        """Determine confidence level based on EOU probability"""
        if eou_prob > 0.8:
            return "high"
        elif eou_prob > 0.6:
            return "medium"
        elif eou_prob > 0.4:
            return "low"
        else:
            return "very_low"
    
    def should_end_turn(self, messages: List[Dict[str, str]], language: str = "en") -> Dict[str, Any]:
        """
        Convenience method to get turn decision with confidence metrics
        
        Args:
            messages: List of conversation messages
            language: Language code for threshold selection
            
        Returns:
            dict: Turn decision with confidence metrics
        """
        result = self.predict_eou(messages)
        
        # Handle prediction errors
        if "error" in result:
            return {
                "should_end": False,
                "eou_probability": 0.0,
                "threshold": self.get_language_threshold(language),
                "confidence": "error",
                "duration_ms": result["duration_ms"],
                "error": result["error"]
            }
        
        # Calculate turn decision
        eou_prob = result["eou_probability"]
        threshold = self.get_language_threshold(language)
        should_end = eou_prob > threshold
        confidence = self._get_confidence_level(eou_prob)
        
        return {
            "should_end": should_end,
            "eou_probability": eou_prob,
            "threshold": threshold,
            "confidence": confidence,
            "duration_ms": result["duration_ms"]
        }