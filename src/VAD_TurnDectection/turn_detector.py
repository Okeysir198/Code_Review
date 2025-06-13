"""
Separate Turn Detectors for LiveKit and TEN models
"""

import time
from typing import List, Dict, Any
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class LiveKitTurnDetector:
    """LiveKit turn detector using ONNX models"""
    
    def __init__(self, language: str = "multilingual"):
        """
        Args:
            language: "multilingual" or "en"
        """
        revisions = {"en": "v1.2.2-en", "multilingual": "v0.2.0-intl"}
        self.revision = revisions[language]
        
        # Load ONNX model
        model_path = hf_hub_download(
            "livekit/turn-detector", "model_q8.onnx",
            subfolder="onnx", revision=self.revision
        )
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "livekit/turn-detector", revision=self.revision, truncation_side="left"
        )
        print(f"✓ LiveKit model loaded ({language})")
    
    def predict_eou(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Predict end-of-utterance probability (original LiveKit method)"""
        start_time = time.perf_counter()
        
        try:
            # Format messages
            text = self._format_messages(messages)
            if not text:
                raise ValueError("No valid content found in messages")
            
            # Tokenize
            inputs = self.tokenizer(
                text, add_special_tokens=False, return_tensors="np",
                max_length=128, truncation=True
            )
            
            # Inference
            outputs = self.session.run(None, {"input_ids": inputs["input_ids"].astype("int64")})
            eou_probability = float(outputs[0].flatten()[-1])
            
            return {
                "eou_probability": eou_probability,
                "input": text,
                "duration_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "raw_output_shape": outputs[0].shape
            }
        except Exception as e:
            return {
                "eou_probability": 0.0,
                "input": "",
                "duration_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "error": str(e)
            }
    
    def predict(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Predict turn probability (simplified interface)"""
        result = self.predict_eou(messages)
        
        if "error" in result:
            return result
        
        eou_probability = result["eou_probability"]
        return {
            "type": "livekit",
            "eou_probability": eou_probability,
            "state": "finished" if eou_probability > 0.5 else "unfinished",
            "duration_ms": result["duration_ms"]
        }
    
    def should_end_turn(self, messages: List[Dict[str, str]], threshold: float = 0.5) -> Dict[str, Any]:
        """Determine if turn should end"""
        result = self.predict(messages)
        
        if "error" in result:
            return {"should_end": False, "confidence": "error", **result}
        
        prob = result["eou_probability"]
        should_end = prob > threshold
        
        
        return {"should_end": should_end, **result}
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for LiveKit"""
        filtered = [msg for msg in messages if msg.get("content", "").strip()]
        if not filtered:
            return ""
        
        text = self.tokenizer.apply_chat_template(
            filtered, add_generation_prompt=False, 
            add_special_tokens=False, tokenize=False
        )
        
        # Remove final EOU token
        end_idx = text.rfind("<|im_end|>")
        return text[:end_idx] if end_idx != -1 else text


class TENTurnDetector:
    """TEN turn detector using Ollama"""
    
    def __init__(self, model_name: str = "hf.co/Mungert/TEN_Turn_Detection-GGUF:Q4_K_M"):
        """
        Args:
            model_name: GGUF model name for Ollama
            e.g: hf.co/Mungert/TEN_Turn_Detection-GGUF:Q4_K_M
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError("langchain_ollama required for TEN model")
        
        self.model = ChatOllama(model=model_name, temperature=0)
        print(f"✓ TEN model loaded: {model_name}")
    
    def predict(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Predict turn state"""
        start_time = time.perf_counter()
        
        try:
            result = self.model.invoke(messages).content.strip().lower()
            
            # Map to standard states
            if result in ["finished", "unfinished", "wait"]:
                state = result
            else:
                state = "unknown"
            
            # Convert to probability for compatibility
            eou_probability = 0.9 if state == "finished" else 0.1
            
            return {
                "type": "ten",
                "state": state,
                "eou_probability": eou_probability,
                "raw_result": result,
                "duration_ms": round((time.perf_counter() - start_time) * 1000, 2)
            }
        except Exception as e:
            return {
                "error": str(e),
                "duration_ms": round((time.perf_counter() - start_time) * 1000, 2)
            }
    
    def should_end_turn(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Determine if turn should end"""
        result = self.predict(messages)
        
        if "error" in result:
            return {"should_end": False, "confidence": "error", **result}
        
        state = result["state"]
        prob = result["eou_probability"]
        
        should_end = state == "finished"
        
        return {"should_end": should_end, **result}


# Usage Examples
if __name__ == "__main__":
    # Test messages
    messages = [
        {"role": "assistant", "content": "How can I help you today?"},
        {"role": "user", "content": "Tell me about yourself. ah, let me think, please say something"}
    ]
    
    # LiveKit example
    try:
        detector = LiveKitTurnDetector(language="multilingual")
        result = detector.should_end_turn(messages)
        print(f"LiveKit: {result}")
    except Exception as e:
        print(f"LiveKit failed: {e}")
    
    # TEN example  
    try:
        detector = TENTurnDetector(model_name="hf.co/Mungert/TEN_Turn_Detection-GGUF:Q4_K_M")
        result = detector.should_end_turn(messages)
        print(f"TEN: {result}")
    except Exception as e:
        print(f"TEN failed: {e}")