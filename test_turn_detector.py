"""
Turn Detection Testing Interface
Clean Gradio UI with Ocean theme using default components only
"""

import gradio as gr
import json
import threading
from typing import List, Dict, Any, Tuple

# Import the turn detector classes
try:
    from src.VAD_TurnDectection.turn_detector import LiveKitTurnDetector, TENTurnDetector
    print("✓ Turn detector classes imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Create dummy classes for demo purposes
    class LiveKitTurnDetector:
        def __init__(self, language="multilingual"):
            self.language = language
        
        def should_end_turn(self, messages, threshold=0.5):
            return {"error": "LiveKit not available - install dependencies"}
    
    class TENTurnDetector:
        def __init__(self, model_name=""):
            self.model_name = model_name
        
        def should_end_turn(self, messages):
            return {"error": "TEN not available - install langchain_ollama"}

# Global detector instances
livekit_detector = None
ten_detector = None
models_loaded = False
loading_status = "🌊 Models loading..."

def load_models():
    """Load both models automatically"""
    global livekit_detector, ten_detector, models_loaded, loading_status
    
    try:
        loading_status = "🔄 Loading LiveKit..."
        livekit_detector = LiveKitTurnDetector(language="multilingual")
        
        loading_status = "🔄 Loading TEN..."
        ten_detector = TENTurnDetector(model_name="hf.co/Mungert/TEN_Turn_Detection-GGUF:Q4_K_M")
        
        models_loaded = True
        loading_status = "✅ Models ready!"
        print(loading_status)
        
    except Exception as e:
        models_loaded = False
        loading_status = f"❌ Error: {str(e)}"
        print(loading_status)

def parse_messages(messages_text: str) -> List[Dict[str, str]]:
    """Parse messages from text input with improved error handling"""
    try:
        # Try JSON format first
        if messages_text.strip().startswith('['):
            return json.loads(messages_text)
        
        # Simple text format: role:content per line
        messages = []
        lines = [line.strip() for line in messages_text.strip().split('\n') if line.strip()]
        
        for line in lines:
            if ':' in line:
                role, content = line.split(':', 1)
                messages.append({
                    "role": role.strip(),
                    "content": content.strip()
                })
        
        return messages if messages else [{"role": "user", "content": messages_text}]
    except Exception:
        # Fallback: treat as single user message
        return [{"role": "user", "content": messages_text}]

def process_turn_detection(messages_text: str, threshold: float) -> Tuple[str, str, str, str, str]:
    """Process both models and return results"""
    if not models_loaded:
        return loading_status, "", "", "", ""
    
    try:
        messages = parse_messages(messages_text)
        
        # Test LiveKit
        livekit_result = {"error": "Not available"}
        if livekit_detector:
            try:
                livekit_result = livekit_detector.should_end_turn(messages, threshold=threshold)
            except Exception as e:
                livekit_result = {"error": str(e)}
        
        # Test TEN
        ten_result = {"error": "Not available"}
        if ten_detector:
            try:
                ten_result = ten_detector.should_end_turn(messages)
            except Exception as e:
                ten_result = {"error": str(e)}
        
        # Format summary
        summary = "# 🌊 Turn Detection Results\n\n"
        
        # LiveKit summary
        if "error" in livekit_result:
            summary += f"## 🤖 LiveKit: ❌ Error\n{livekit_result['error']}\n\n"
            livekit_status = "❌ Error"
            livekit_details = f"Probability: N/A\nDuration: N/A"
        else:
            prob = livekit_result.get('eou_probability', 0)
            should_end = livekit_result.get('should_end', False)
            duration = livekit_result.get('duration_ms', 0)
            
            decision = "✅ END TURN" if should_end else "❌ CONTINUE"
            summary += f"## 🤖 LiveKit: {decision}\n"
            summary += f"- Probability: {prob:.3f}\n"
            summary += f"- Threshold: {threshold}\n"
            summary += f"- Duration: {duration}ms\n\n"
            
            livekit_status = decision
            livekit_details = f"Probability: {prob:.3f}\nThreshold: {threshold}\nDuration: {duration}ms"
        
        # TEN summary
        if "error" in ten_result:
            summary += f"## 🎯 TEN: ❌ Error\n{ten_result['error']}\n\n"
            ten_status = "❌ Error"
            ten_details = f"State: N/A\nDuration: N/A"
        else:
            should_end = ten_result.get('should_end', False)
            state = ten_result.get('state', 'unknown')
            duration = ten_result.get('duration_ms', 0)
            
            decision = "✅ END TURN" if should_end else "❌ CONTINUE"
            summary += f"## 🎯 TEN: {decision}\n"
            summary += f"- State: {state}\n"
            summary += f"- Duration: {duration}ms\n\n"
            
            ten_status = decision
            ten_details = f"State: {state}\nDuration: {duration}ms"
        
        # Agreement check
        livekit_decision = livekit_result.get('should_end', False) if "error" not in livekit_result else None
        ten_decision = ten_result.get('should_end', False) if "error" not in ten_result else None
        
        if livekit_decision is not None and ten_decision is not None:
            if livekit_decision == ten_decision:
                summary += "## 🎯 Agreement: ✅ Both models agree!\n"
            else:
                summary += "## ⚠️ Disagreement: Models have different decisions\n"
        
        # Format JSON outputs
        livekit_json = json.dumps(livekit_result, indent=2)
        ten_json = json.dumps(ten_result, indent=2)
        
        return summary, livekit_status, livekit_details, ten_status, ten_details
        
    except Exception as e:
        error_msg = f"❌ Processing failed: {str(e)}"
        return error_msg, "", "", "", ""

def get_finished_examples() -> List[str]:
    """Get examples that represent finished utterances from STT"""
    return [
        "assistant:how can i help you today\nuser:can you tell me the weather forecast for tomorrow.",
        "assistant:what would you like to know\nuser:i need help setting up my email account.",
        "assistant:how may i assist you\nuser:please show me how to reset my password.",
        "assistant:what brings you here today\nuser:i want to learn about machine learning basics.", 
        "assistant:what can i do for you\nuser:help me understand python programming.",
        "assistant:how are you doing\nuser:im doing great thanks for asking.",
        "assistant:what do you need help with\nuser:i need to schedule a meeting for next week.",
        "assistant:anything else i can help with\nuser:no thats everything thank you very much.",
        "assistant:whats your question\nuser:how do i create a new database table.",
        "assistant:tell me more\nuser:i want to build a web application using react.",
        "assistant:what would you like to discuss\nuser:can you explain how artificial intelligence works.",
        "assistant:how can i assist\nuser:i need directions to the nearest coffee shop."
    ]

def get_unfinished_examples() -> List[str]:
    """Get examples that represent unfinished utterances from STT"""
    return [
        "assistant:what can i help you with\nuser:well i was thinking maybe you could",
        "assistant:tell me what you need\nuser:so basically what i want to do is",
        "assistant:how may i assist you\nuser:um let me see i need help with",
        "assistant:what would you like to know\nuser:i was wondering if you could help me",
        "assistant:whats on your mind\nuser:the thing is when i try to use",
        "assistant:how can i help\nuser:so i have this problem where",
        "assistant:what brings you here\nuser:well its kind of complicated but",
        "assistant:tell me more about it\nuser:okay so what happens is when i",
        "assistant:what do you need\nuser:i think i need some assistance with",
        "assistant:how are you\nuser:im doing okay but i was hoping",
        "assistant:what can i do for you\nuser:let me think about how to explain",
        "assistant:whats your question\nuser:its hard to describe but basically"
    ]

def create_turn_detection_demo():
    """Create and return the turn detection demo interface"""
    
    # Load models in background
    threading.Thread(target=load_models, daemon=True).start()
    
    with gr.Blocks(theme=gr.themes.Ocean(), title="Turn Detection Tester") as demo:
        
        gr.Markdown("# 🌊 Turn Detection Tester")
        gr.Markdown("Compare LiveKit and TEN turn detection models side-by-side")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Input")
                
                messages_input = gr.Textbox(
                    lines=8,
                    placeholder="Enter messages in 'role:content' format or JSON...",
                    label="Conversation Messages",
                    value="assistant:what can i help you with\nuser:well i was thinking maybe you could"
                )
                
                threshold_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="LiveKit Threshold"
                )
                
                process_btn = gr.Button(
                    "🚀 Analyze Turn Detection",
                    variant="primary",
                    size="lg"
                )
                # # Detailed results
                gr.Markdown("### 📋 Detailed Summary")
                results_summary = gr.Markdown("Click 'Analyze Turn Detection' to see detailed results...")
                
            
            with gr.Column(scale=3):
                gr.Markdown("### 📊 Results Overview")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🤖 LiveKit")
                        livekit_status = gr.Textbox(
                            label="Decision",
                            value="Click analyze to see results",
                            interactive=False
                        )
                        livekit_info = gr.Textbox(
                            label="Details",
                            lines=3,
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("#### 🎯 TEN")
                        ten_status = gr.Textbox(
                            label="Decision", 
                            value="Click analyze to see results",
                            interactive=False
                        )
                        ten_info = gr.Textbox(
                            label="Details",
                            lines=3,
                            interactive=False
                        )
                # Examples section with two columns
                gr.Markdown("### 📚 STT Example Tests")
                gr.Markdown("Click examples to test how models handle typical Speech-to-Text outputs:")
                with gr.Row():
                    
                    with gr.Column():
                        gr.Markdown("#### ✅ Finished Utterances")
                        finished_examples = get_finished_examples()
                        
                        for example_text in finished_examples:
                            # Create short preview for button
                            user_part = example_text.split('\nuser:')[1] if '\nuser:' in example_text else example_text
                            preview = user_part[:45] + ("" if len(user_part) <= 45 else "")
                            
                            example_btn = gr.Button(preview, size="sm")
                            example_btn.click(
                                fn=lambda text=example_text: text,
                                outputs=[messages_input]
                            )
                    
                    with gr.Column():
                        gr.Markdown("#### ❌ Unfinished Utterances")
                        unfinished_examples = get_unfinished_examples()
                        
                        for example_text in unfinished_examples:
                            # Create short preview for button
                            user_part = example_text.split('\nuser:')[1] if '\nuser:' in example_text else example_text
                            preview = user_part[:45] + ("" if len(user_part) <= 45 else "")
                            
                            example_btn = gr.Button(preview, size="sm")
                            example_btn.click(
                                fn=lambda text=example_text: text,
                                outputs=[messages_input]
                            )
        
        
        
        
        
        
        # Raw outputs (collapsible)
        with gr.Accordion("🔍 Raw Model Outputs", open=False, visible=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### LiveKit JSON")
                    livekit_raw = gr.JSON(label="Raw Response")
                
                with gr.Column():
                    gr.Markdown("#### TEN JSON")
                    ten_raw = gr.JSON(label="Raw Response")
        
        # Event handler
        process_btn.click(
            fn=process_turn_detection,
            inputs=[messages_input, threshold_slider],
            outputs=[results_summary, livekit_status, livekit_info, ten_status, ten_info]
        )
        
        # # Also update raw outputs
        # process_btn.click(
        #     fn=lambda msg, thresh: (
        #         json.loads(process_turn_detection(msg, thresh)[4]) if process_turn_detection(msg, thresh)[4] else {},
        #         json.loads(process_turn_detection(msg, thresh)[5]) if process_turn_detection(msg, thresh)[5] else {}
        #     ),
        #     inputs=[messages_input, threshold_slider], 
        #     outputs=[livekit_raw, ten_raw]
        # )
    
    return demo

# Main execution
if __name__ == "__main__":
    demo = create_turn_detection_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )