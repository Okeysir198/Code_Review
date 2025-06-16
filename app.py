"""
Tabbed Interface Launcher
Tab 1: Real-time Audio Processing (from paste-2.txt)
Tab 2: Noise Reduction Testing (from paste.txt)
"""

import gradio as gr

# Import the interfaces from the existing files
from simple_fastrtc import create_gradio_interface as create_realtime_interface
from noise_reduction_test_ui import create_interface as create_noise_testing_interface
from test_turn_detector import create_turn_detection_demo

def create_combined_interface():
    """Create combined tabbed interface"""
    
    with gr.Blocks(
        title="Audio Processing Suite",
        theme=gr.themes.Ocean(),
        fill_width=True,
    ) as demo:
        
        gr.HTML("""
        <h1 style='text-align: center; color: #2563eb;'>
            🎤 Audio Processing Suite
        </h1>
        """)
        
        with gr.Tabs():
            with gr.Tab("🚀 Test Background Noise Reduction"):
                noise_demo = create_noise_testing_interface()
            with gr.Tab("🚀 Test semantic turn detection (detect user finishes talking)"):
                turn_detector_demo = create_turn_detection_demo()
            with gr.Tab("🎙️ Test interruption"):
                realtime_demo = create_realtime_interface()
                # realtime_demo.render()
            
            
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting Combined Audio Processing Suite...")
    print("Tab 1: Real-time Audio Processing with STT/TTS")
    print("Tab 2: Noise Reduction Testing Interface")
    
    demo = create_combined_interface()
    demo.launch(
        share=True,
        server_port=7860,
        server_name="0.0.0.0",
        show_error=True
    )