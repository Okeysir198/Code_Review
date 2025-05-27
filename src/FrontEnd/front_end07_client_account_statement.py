import gradio as gr
import pandas as pd
import sys
import datetime
sys.path.append("../..")
from src.Database import CartrackSQLDatabase

def display_account_statement(user_id=None):
    """Display client account statement using DataFrame + summary"""
    if not user_id:
        return None, "Please enter a client ID and click Search to view account statement."
    
    data = CartrackSQLDatabase.get_client_account_statement.invoke(user_id)
    if not data:
        return None, f"❌ **No data found for client ID: {user_id}**"
    
    df = pd.DataFrame(data)
    if df.empty:
        return None, f"❌ **No transactions found for client ID: {user_id}**"
    
    # Process DataFrame for display
    df_display = df.copy()
    df_display['Date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
    df_display['Time'] = pd.to_datetime(df['timestamp']).dt.strftime('%H:%M')
    df_display['Type'] = df['doc_type'].map({'Payment': '💰 Payment', 'Invoice': '📄 Invoice'}).fillna('📋 ' + df['doc_type'])
    df_display['Status'] = df['amount'].astype(float).apply(lambda x: '🟢 Credit' if x < 0 else '🔴 Debit')
    df_display['Amount'] = df['amount'].astype(float).apply(lambda x: f"R {x:,.2f}")
    df_display['Outstanding'] = df['amount_outstanding'].astype(float).apply(lambda x: f"R {x:,.2f}")
    df_display['Description'] = df['comment'].fillna('N/A').apply(lambda x: x[:40] + '...' if len(str(x)) > 40 else x)
    
    # Select and rename columns
    df_final = df_display[['Date', 'Time', 'Type', 'Status', 'Description', 'Amount', 'Outstanding', 'invoice_hdr_id']].rename(columns={
        'invoice_hdr_id': 'Reference'
    })
    
    # Calculate summary
    invoices = len(df[df['amount'].astype(float) > 0])
    payments = len(df[df['amount'].astype(float) < 0])
    total_outstanding = df['amount_outstanding'].astype(float).sum()
    
    summary = f"""
# 💳 **Account Statement - Client {user_id}**
📊 **Summary:** {invoices} Invoices • {payments} Payments • **R {total_outstanding:,.2f}** Outstanding
📅 **Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
"""
    
    return df_final, summary

def create_account_statement_app():
    with gr.Blocks(title="💳 Account Statement") as app:
        gr.Markdown("# 💳 **Account Statement Dashboard**")
        
        with gr.Row():
            user_id_input = gr.Textbox(label="🆔 Client ID", value="28173", scale=2)
            search_button = gr.Button("🚀 Search", variant="primary", scale=1)
        
        summary_output = gr.Markdown()
        dataframe_output = gr.Dataframe(
            headers=["Date", "Time", "Type", "Status", "Description", "Amount", "Outstanding", "Reference"],
            max_height=500,
            show_search='filter',
            wrap=False
        )
        
        def update_display(user_id):
            df, summary = display_account_statement(user_id)
            return summary, df
        
        search_button.click(update_display, user_id_input, [summary_output, dataframe_output])
        user_id_input.submit(update_display, user_id_input, [summary_output, dataframe_output])
        
        # Load initial data
        initial_df, initial_summary = display_account_statement("28173")
        summary_output.value = initial_summary
        dataframe_output.value = initial_df
    
    return app

# if __name__ == "__main__":
#     demo = create_account_statement_app()
#     demo.launch()