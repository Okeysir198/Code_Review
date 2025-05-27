import gradio as gr
import pandas as pd
import sys
import datetime
sys.path.append("../..")
from src.Database import CartrackSQLDatabase

def display_client_contracts(user_id=None):
    """Display client contracts using DataFrame + summary"""
    if not user_id:
        return None, "Please enter a client ID and click Search to view contracts."
    
    contracts = CartrackSQLDatabase.get_client_contracts.invoke(user_id)
    if not contracts:
        return None, f"❌ **No contracts found for client ID: {user_id}**"
    
    df = pd.DataFrame(contracts)
    if df.empty:
        return None, f"❌ **No contract data for client ID: {user_id}**"
    
    # Process DataFrame for display
    df_display = df.copy()
    
    # Format key columns
    df_display['Contract ID'] = df['contract_id'].fillna('N/A')
    df_display['Status'] = df['contract_state'].fillna('Unknown').apply(lambda x: f"{get_status_emoji(x)} {x}")
    df_display['Sale Date'] = pd.to_datetime(df['sale_date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('N/A')
    df_display['Start Date'] = pd.to_datetime(df['contract_start_date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('N/A')
    df_display['End Date'] = pd.to_datetime(df['contract_end_date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('Ongoing')
    df_display['Payment Option'] = df['payment_option'].fillna('N/A')
    df_display['Vehicle ID'] = df['vehicle_id'].fillna('N/A')
    df_display['Terminal ID'] = df['terminal_id'].fillna('N/A')
    df_display['Fitter ID'] = df['fitter_id'].fillna('N/A')
    df_display['Package ID'] = df['product_package_id'].fillna('N/A')
    df_display['Branch ID'] = df['branch_id'].fillna('N/A')
    
    # Select final columns
    df_final = df_display[[
        'Contract ID', 'Status', 'Sale Date', 'Start Date', 'End Date', 
        'Payment Option', 'Vehicle ID', 'Terminal ID', 'Fitter ID', 'Package ID', 'Branch ID'
    ]]
    
    # Calculate summary stats
    total_contracts = len(df)
    active_contracts = len(df[df['contract_state'].str.contains('active', case=False, na=False)])
    cancelled_contracts = len(df[df['contract_state'].str.contains('cancel', case=False, na=False)])
    payment_options = df['payment_option'].dropna().unique()
    
    # Date range
    earliest_date = pd.to_datetime(df['contract_start_date'], errors='coerce').min()
    latest_date = pd.to_datetime(df['contract_start_date'], errors='coerce').max()
    date_range = f"{earliest_date.strftime('%Y-%m-%d') if pd.notna(earliest_date) else 'N/A'} to {latest_date.strftime('%Y-%m-%d') if pd.notna(latest_date) else 'N/A'}"
    
    summary = f"""
# 📋 **Client Contracts - Client {user_id}**

📊 **Summary:** {total_contracts} Total • {active_contracts} Active • {cancelled_contracts} Cancelled  
💳 **Payment:** {', '.join(payment_options) if len(payment_options) > 0 else 'N/A'}  
📅 **Date Range:** {date_range}  
📅 **Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
"""
    
    return df_final, summary

def get_status_emoji(status):
    """Get emoji for contract status"""
    if not status:
        return "❓"
    status_lower = str(status).lower()
    if "active" in status_lower:
        return "✅"
    elif "cancel" in status_lower:
        return "❌"
    elif "pending" in status_lower:
        return "⏳"
    elif "repaired" in status_lower:
        return "🔧"
    else:
        return "ℹ️"

def create_client_contracts_app():
    with gr.Blocks(title="📋 Client Contracts") as app:
        gr.Markdown("# 📋 **CLIENT CONTRACTS DASHBOARD**")
        
        with gr.Row():
            user_id_input = gr.Textbox(label="🆔 Client ID", value="28173", scale=2)
            search_button = gr.Button("🚀 Search", variant="primary", scale=1)
        
        summary_output = gr.Markdown()
        dataframe_output = gr.Dataframe(
            headers=["Contract ID", "Status", "Sale Date", "Start Date", "End Date", 
                    "Payment Option", "Vehicle ID", "Terminal ID", "Fitter ID", "Package ID", "Branch ID"],
            wrap=False,
            show_search='filter'
        )
        
        def update_display(user_id):
            df, summary = display_client_contracts(user_id)
            return summary, df
        
        search_button.click(update_display, user_id_input, [summary_output, dataframe_output])
        user_id_input.submit(update_display, user_id_input, [summary_output, dataframe_output])
        
        # Load initial data
        initial_df, initial_summary = display_client_contracts("28173")
        summary_output.value = initial_summary
        dataframe_output.value = initial_df
    
    return app

# if __name__ == "__main__":
#     demo = create_client_contracts_app()
#     demo.launch()