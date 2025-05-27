import gradio as gr
import pandas as pd
import sys
import datetime
sys.path.append("../..")
from src.Database import CartrackSQLDatabase

def display_client_profile(user_id=None):
    """Display client profile information using markdown + DataFrame for vehicles"""
    if not user_id:
        return None, "Please enter a client ID and click Search to view client profile details."
    
    # Get client profile data
    data = CartrackSQLDatabase.get_client_profile.invoke(user_id)
    
    if not data:
        return None, f"❌ **Client profile not found**\n\nNo profile information available for client ID: {user_id}"
    
    # Extract data
    client_info = data.get('client_info', {})
    addresses = data.get('addresses', [])
    vehicles = data.get('vehicles', [])
    sim_card = data.get('sim_card', {})
    fitment = data.get('fitment', {})
    billing = data.get('billing', {})
    
    # Build address information with better formatting
    address_info = "🚫 *No address information available*"
    if addresses:
        address_lines = []
        for address in addresses:
            lines = [
                f"🏠 **{address.get('type', 'Address')} Address:** ",
                f"📍 `{address.get('address_line1', '')}`",
                f"📍 `{address.get('address_line2', '')}`" if address.get('address_line2') else "",
                f"🌍 `{address.get('province', '')}, {address.get('post_code', '')}`"
            ]
            address_info = "<br><br>".join([line for line in lines if line.strip()])
    
    # Create vehicles DataFrame
    df_vehicles = pd.DataFrame()
    if vehicles:
        vehicle_data = []
        for vehicle in vehicles:
            status_emoji = "🟢" if vehicle.get('contract_status', '').lower() == 'active' else "🔴"
            status = f"{status_emoji} {vehicle.get('contract_status', 'Unknown')}"
            make_model = f"🚙 {vehicle.get('make', 'Unknown')} {vehicle.get('model', '')}"
            registration = vehicle.get('registration', 'N/A')
            color = f"🎨 {vehicle.get('color', 'N/A')}"
            year = vehicle.get('model_year', 'N/A')
            chassis = vehicle.get('chassis_number', 'N/A')
            terminal_serial = vehicle.get('terminal_serial', 'N/A')
            terminal_response = vehicle.get('terminal_last_response', 'N/A')
            
            vehicle_data.append([
                status, make_model, registration, color, year, 
                chassis, terminal_serial, terminal_response
            ])
        
        df_vehicles = pd.DataFrame(vehicle_data, columns=[
            'Status', 'Vehicle', 'Registration', 'Color', 'Year',
            'Chassis', 'Terminal', 'Last Signal'
        ])
    
    # Build markdown output (without vehicles section)
    markdown_output = f"""
# 🏢 **{client_info.get('client_full_name', 'Unknown Client')}**

> 👤 **User Name:**  `{data.get('user_name', 'N/A')}` | 📄 **ID Number:**  `{client_info.get('id_number', 'N/A')}`

## 📊 **CLIENT INFORMATION**

| 👤 **Personal Details** | 📞 **Contact Details** | 🏠 **Address Details** | 💳 **Billing Details** | ⚙️ **Technical Details** |
|:---|:---|:---|:---|:---|
| 🏷️ **Title:** `{client_info.get('title', 'N/A')}`<br><br>👨‍💼 **Name:** `{client_info.get('first_name', 'N/A')} {client_info.get('last_name', 'N/A')}`<br><br>🆔 **ID Number:** `{client_info.get('id_number', 'N/A')}`<br><br>🛂 **Passport:** `{client_info.get('passport') or 'None'}`<br><br>🎂 **Date of Birth:** `{client_info.get('date_of_birth_raw', 'N/A')}`<br><br>🏢 **Employer:** `{client_info.get('employer') or 'None'}` | 📱 **Mobile:** `{client_info.get('contact', {}).get('mobile', 'N/A')}`<br><br>☎️ **Telephone:** `{client_info.get('contact', {}).get('telephone', 'N/A')}`<br><br>📠 **Fax:** `{client_info.get('contact', {}).get('fax', 'N/A')}`<br><br>📧 **Email:** `{client_info.get('email_address', 'N/A')}`{4*'<br>'} | {address_info} {4*'<br>'}| 📦 **Package:** `{billing.get('product_package') or 'None'}`<br><br>🔄 **Recovery:** `{billing.get('recovery_option') or 'None'}` {8*'<br>'}| 📅 **Fitment Date:** `{fitment.get('date') or 'None'}`<br><br>👷 **Fitter:** `{fitment.get('fitter_name') or 'None'}`<br><br>📡 **SIM ID:** `{sim_card.get('chip_id') or 'None'}`{6*'<br>'}|

---

### 📋 **Export Information**
> 📅 **Generated:**  `{datetime.datetime.now().strftime("%B %d, %Y")}` | ⏰ **Time:**  `{datetime.datetime.now().strftime("%I:%M:%S %p")}` | 🚗 **Vehicles:** `{len(vehicles)} found`

"""
    
    return df_vehicles, markdown_output

def create_client_profile_app():
    with gr.Blocks(title="🔍 Client Profile Dashboard") as app:
        gr.Markdown("# 🎯 **CLIENT PROFILE DASHBOARD**")
        gr.Markdown("### 🔍 Search and explore comprehensive client information")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 🔎 **Search Panel**")
                user_id_input = gr.Textbox(
                    label="🆔 Client ID", 
                    placeholder="Enter client ID...",
                    value="28173"
                )
                search_button = gr.Button("🚀 Search Client", variant="primary", size="lg")
            
            with gr.Column(scale=4):
                gr.Markdown("#### 📊 **Profile Results**")
                profile_markdown = gr.Markdown(
                    elem_id="profile-output"
                )
                
                gr.Markdown("### 🚗 **VEHICLE FLEET**")
                vehicles_dataframe = gr.Dataframe(
                    headers=["Status", "Vehicle", "Registration", "Color", "Year", "Chassis", "Terminal", "Last Signal"],
                    wrap=False,
                    interactive=False
                )
        
        def update_profile(user_id):
            df_vehicles, markdown = display_client_profile(user_id)
            return markdown, df_vehicles
        
        # Event handlers
        search_button.click(
            fn=update_profile, 
            inputs=user_id_input, 
            outputs=[profile_markdown, vehicles_dataframe]
        )
        
        user_id_input.submit(
            fn=update_profile, 
            inputs=user_id_input, 
            outputs=[profile_markdown, vehicles_dataframe]
        )
        
        # Load initial data
        initial_df, initial_markdown = display_client_profile("28173")
        profile_markdown.value = initial_markdown
        vehicles_dataframe.value = initial_df
    
    return app

# if __name__ == "__main__":
#     demo = create_client_profile_app()
#     demo.launch()