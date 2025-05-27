import sys
import logging
import base64
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import gradio as gr

# Setup
load_dotenv(find_dotenv())
sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from app_config import CONFIG
from src.FrontEnd import create_client_overview_interface, create_client_details_interface
from src.FrontEnd import create_audio_processing_tab

# CSS for layout
custom_css = """
/* Hide footer*/
footer {
  visibility: hidden;
}

/* Modern Voice Chat Floating Panel */
.voice-chat-floating {
    position: fixed !important;
    top: 24px !important;
    right: 24px !important;
    width: 380px !important;
    max-width: calc(100vw - 48px) !important;
    z-index: 1000 !important;
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(226, 232, 240, 0.8) !important;
    border-radius: 16px !important;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04) !important;
    padding: 10px;
}

.voice-chat-floating:hover {
    box-shadow: 0 25px 30px -5px rgba(0, 0, 0, 0.15), 0 15px 15px -5px rgba(0, 0, 0, 0.06) !important;
}

.main-content {
    margin-right: 400px !important;
    padding: 20px !important;
}

@media (max-width: 1200px) {
    .voice-chat-floating { 
        position: relative !important;
        top: auto !important;
        right: auto !important; 
        width: 100% !important; 
        margin: 20px 0 !important;
        backdrop-filter: none !important;
        background: white !important;
        transform: none !important;
    }
    .voice-chat-floating:hover {
        transform: none !important;
    }
    .main-content { 
        margin-right: 0 !important; 
    }
}

@media (prefers-color-scheme: dark) {
    .voice-chat-floating {
        background: rgba(15, 23, 42, 0.95) !important;
        border-color: rgba(51, 65, 85, 0.8) !important;
    }
}
"""

def get_logo_base64():
    """Convert logo to base64 for embedding in HTML"""
    try:
        logo_path = Path(__file__).parent / "asset" / "cartrack_logo.svg"
        if logo_path.exists():
            with open(logo_path, 'rb') as f:
                logo_data = f.read()
                logo_base64 = base64.b64encode(logo_data).decode()
                
                # Determine MIME type
                mime_type = {
                    '.svg': "image/svg+xml",
                    '.png': "image/png", 
                    '.jpg': "image/jpeg",
                    '.jpeg': "image/jpeg"
                }.get(logo_path.suffix.lower(), "image/svg+xml")
                
                return f"data:{mime_type};base64,{logo_base64}"
    except Exception as e:
        print(f"Error loading logo: {e}")
    return None

def create_header(logo_data_uri):
    """Create app header with logo"""
    if logo_data_uri:
        return f"""
            <div style="display: flex; align-items: center; padding: 15px 20px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        color: white; border-radius: 12px; margin-bottom: 20px;">
                
                <!-- Logo container -->
                <div style="background: white; border-radius: 8px; padding: 0px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); 
                            display: flex; align-items: center; justify-content: center; min-width: 180px; height: 60px; margin-right: 20px;">
                    <img src="{logo_data_uri}" alt="Company Logo" 
                         style="max-width: 100%; max-height: 100%; object-fit: contain;" />
                </div>
                
                <!-- Main content -->
                <div style="flex: 1; text-align: center;">
                    <h3 style="margin: 0 0 8px 0; font-size: 24px; font-weight: 600;">📞 Call Center AI Agent</h3>
                    <p style="margin: 0; opacity: 0.9; font-size: 16px;">Professional Debt Management & Client Information System</p>
                </div>
            </div>
        """
    else:
        return """
            <div style="display: flex; align-items: center; padding: 15px 20px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        color: white; border-radius: 12px; margin-bottom: 20px;">
                
                <!-- Logo placeholder -->
                <div style="background: white; border-radius: 8px; padding: 12px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); 
                            display: flex; align-items: center; justify-content: center; min-width: 80px; height: 60px; margin-right: 20px;">
                    <div style="color: #10b981; font-weight: bold; font-size: 14px; text-align: center;">LOGO</div>
                </div>
                
                <!-- Main content -->
                <div style="flex: 1; text-align: center;">
                    <h3 style="margin: 0 0 8px 0; font-size: 24px; font-weight: 600;">📞 Call Center AI Agent</h3>
                    <p style="margin: 0; opacity: 0.9; font-size: 16px;">Professional Debt Management & Client Information System</p>
                </div>
            </div>
        """

def create_main_app():
    """Create the main application"""
    theme = gr.themes.Base(primary_hue="emerald", secondary_hue="sky")
    logo_data_uri = get_logo_base64()
    
    with gr.Blocks(theme=theme, title="📞 Call Center AI Agent", css=custom_css, fill_width=True) as app:
        # Header
        gr.HTML(create_header(logo_data_uri))
       
        # Main tabs
        with gr.Tabs() as tabs:
            with gr.Tab("📊 Client Overview", id=0):
                view_profile_btn, selected_client_id = create_client_overview_interface()

            with gr.Tab("👤 Client Details", id=1):
                client_details_interface, user_id_input = create_client_details_interface()

            with gr.Tab("⚙️ Settings", id=2, visible=False):
                create_settings_panel()

            create_audio_processing_tab(config=CONFIG, visible=True)
        # Event handlers
        def navigate_to_client_details(client_id):
            if client_id:
                return gr.update(selected=1), str(client_id)
            return gr.update(), ""
        
        view_profile_btn.click(
            fn=navigate_to_client_details,
            inputs=[selected_client_id],
            outputs=[tabs, user_id_input]
        )
    
    return app

def create_settings_panel():
    """Create settings panel"""
    gr.Markdown("## ⚙️ System Settings")
    
    with gr.Accordion("🗄️ Database", open=False):
        with gr.Row():
            db_status = gr.Markdown("🟢 **Status:** Connected")
            test_db_btn = gr.Button("🔍 Test Connection", size="sm")
        db_timeout = gr.Number(label="Timeout (seconds)", value=30, minimum=5, maximum=300)
    
    with gr.Accordion("🎙️ Voice Assistant", open=False):
        voice_enabled = gr.Checkbox(label="Enable Voice", value=True)
        voice_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.35, label="Voice Speed")
    
    with gr.Accordion("🎨 Interface", open=True):
        with gr.Row():
            theme_selection = gr.Radio(choices=["Light", "Dark", "Auto"], value="Light", label="Theme")
            results_per_page = gr.Slider(minimum=10, maximum=50, value=20, label="Results Per Page")
        auto_refresh = gr.Checkbox(label="Auto-refresh Data", value=False)
    
    with gr.Row():
        save_btn = gr.Button("💾 Save Settings", variant="primary")
        reset_btn = gr.Button("🔄 Reset", variant="secondary")
    
    settings_status = gr.Markdown("Ready...")
    
    # Event handlers
    save_btn.click(lambda: "✅ Settings saved!", outputs=[settings_status])
    reset_btn.click(lambda: "🔄 Reset complete", outputs=[settings_status])
    test_db_btn.click(lambda: "🟢 **Status:** Connection successful", outputs=[db_status])

def create_demo_app():
    """Create demo version"""
    with gr.Blocks(title="📞 Call Center AI Agent - Demo", fill_width=True) as app:
        gr.HTML("""
            <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                        color: white; border-radius: 12px; margin-bottom: 20px;">
                <h2>📞 Call Center AI Agent - Demo Mode</h2>
                <p>⚠️ Running with sample data</p>
            </div>
        """)
        
        with gr.Tabs():
            with gr.Tab("🏠 Demo Overview"):
                gr.Markdown("## Demo Version\n- ✅ UI functional\n- ⚠️ Sample data only")
                
                sample_data = [
                    ["John Doe", "12345 | jdoe", "R 2,500.00", "45 days", "2", "🟡 Payment Plan"],
                    ["Jane Smith", "67890 | jsmith", "R 1,850.75", "30 days", "1", "🟢 Current"],
                    ["Bob Johnson", "13579 | bjohnson", "R 4,200.00", "60 days", "3", "🔴 Overdue"],
                ]
                
                gr.Dataframe(
                    headers=["Name", "ID | Username", "Balance", "Age", "Vehicles", "Status"],
                    value=sample_data,
                    interactive=False
                )
            
            with gr.Tab("🔧 Setup Help"):
                gr.Markdown("""
                ## Setup Instructions
                
                **File Structure:**
                ```
                your_app/
                ├── llm_app.py
                └── asset/
                    └── cartrack_logo.svg
                ```
                
                **Configuration:**
                ```python
                CONFIG = {
                    "database": {"host": "localhost", "port": 5432},
                    "server": {"port": 7899, "host": "0.0.0.0"}
                }
                ```
                """)
    return app

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Call Center AI Agent")
    parser.add_argument("--port", type=int, default=CONFIG.get('server', {}).get('port', 7899))
    parser.add_argument("--host", default=CONFIG.get('server', {}).get('host', '0.0.0.0'))
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    args = parser.parse_args()
    
    print(f"🚀 Starting Call Center AI Agent on {args.host}:{args.port}")
    
    try:
        app = create_demo_app() if args.demo else create_main_app()
        app.launch(
            server_name=args.host, 
            server_port=args.port, 
            share=args.share, 
            inbrowser=True
        )
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("🎭 Falling back to demo mode...")
        create_demo_app().launch(
            server_name=args.host, 
            server_port=args.port, 
            share=args.share
        )

if __name__ == "__main__":
    main()