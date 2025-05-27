import gradio as gr
import pandas as pd
import sys
import datetime
sys.path.append("../..")
from src.Database import CartrackSQLDatabase

def display_payment_history(user_id=None):
    """Display comprehensive client payment history using DataFrame + summary + additional payment data"""
    if not user_id:
        return None, None, None, "Please enter a client ID and click Search to view payment history."
    
    # Helper function to calculate days ago with timezone handling
    def calculate_days_ago(date_str):
        try:
            date_obj = pd.to_datetime(date_str)
            # If timezone-aware, convert to naive datetime
            if date_obj.tz is not None:
                date_obj = date_obj.tz_localize(None)
            return (datetime.datetime.now() - date_obj).days
        except:
            return "Unknown"
    
    # Get main payment history data
    data = CartrackSQLDatabase.get_client_payment_history.invoke(user_id)
    if not data:
        return None, None, None, f"❌ **No payment history found for client ID: {user_id}**"
    
    df = pd.DataFrame(data)
    if df.empty:
        return None, None, None, f"❌ **No payment arrangements found for client ID: {user_id}**"
    
    # Get additional payment data
    failed_payments_data = CartrackSQLDatabase.get_client_failed_payments.invoke(user_id)
    last_successful_payment = CartrackSQLDatabase.get_client_last_successful_payment.invoke(user_id)
    last_valid_payment = CartrackSQLDatabase.get_client_last_valid_payment.invoke(user_id)
    
    # Try to get last reversed payment (might not work yet according to comment)
    try:
        last_reversed_payment = CartrackSQLDatabase.get_client_last_reversed_payment.invoke(user_id)
    except:
        last_reversed_payment = None
    # Process DataFrame for display
    df_display = df.copy()
    
    # Format dates
    df_display['Created Date'] = pd.to_datetime(df['create_ts']).dt.strftime('%Y-%m-%d')
    df_display['Created Time'] = pd.to_datetime(df['create_ts']).dt.strftime('%H:%M')
    df_display['Payment Date'] = pd.to_datetime(df['pay_date']).dt.strftime('%Y-%m-%d')
    df_display['Follow-up Date'] = pd.to_datetime(df['followup_ts']).dt.strftime('%Y-%m-%d')
    
    # Format status with emojis
    status_map = {
        'Failed': '🔴 Failed',
        'Successful': '🟢 Successful',
        'Pending': '🟡 Pending',
        'Cancelled': '⚫ Cancelled',
        'Processing': '🔵 Processing'
    }
    df_display['Status'] = df['arrangement_state'].map(status_map).fillna('❓ ' + df['arrangement_state'].astype(str))
    
    # Format state type
    state_type_map = {
        'Closed': '🔒 Closed',
        'Open': '🔓 Open',
        'Active': '✅ Active',
        'Inactive': '❌ Inactive'
    }
    df_display['State Type'] = df['arrangement_state_type'].map(state_type_map).fillna('📋 ' + df['arrangement_state_type'].astype(str))
    
    # Format payment type
    payment_type_map = {
        'Manual Debit Order': '🏦 Manual Debit',
        'Automatic Debit Order': '🔄 Auto Debit',
        'Credit Card': '💳 Credit Card',
        'Cash': '💵 Cash',
        'Cheque': '📝 Cheque',
        'EFT': '💸 EFT',
        'Online Payment': '🌐 Online'
    }
    df_display['Payment Type'] = df['arrangement_pay_type'].map(payment_type_map).fillna('💰 ' + df['arrangement_pay_type'].astype(str))
    
    # Format amount
    df_display['Amount'] = df['amount'].astype(float).apply(lambda x: f"R {x:,.2f}")
    
    # Format created by
    creator_map = {
        'ai_agent': '🤖 AI Agent',
        'system': '⚙️ System',
        'admin': '👤 Admin',
        'user': '👥 User'
    }
    df_display['Created By'] = df['created_by'].map(creator_map).fillna('👤 ' + df['created_by'].astype(str))
    
    # Format note/description (truncate if too long)
    df_display['Description'] = df['note'].fillna('N/A').apply(
        lambda x: x[:50] + '...' if len(str(x)) > 50 else str(x)
    )
    
    # Add additional info column
    additional_info = []
    for _, row in df.iterrows():
        info_parts = []
        if row['mandate_fee']:
            info_parts.append(f"Fee: R{row['mandate_fee']}")
        if row['cheque_number']:
            info_parts.append(f"Cheque: {row['cheque_number']}")
        if row['online_payment_reference_id_original']:
            info_parts.append(f"Ref: {row['online_payment_reference_id_original']}")
        additional_info.append(" | ".join(info_parts) if info_parts else "N/A")
    
    df_display['Additional Info'] = additional_info
    
    # Select and organize columns for final display
    df_final = df_display[[
        'Created Date', 'Created Time', 'Payment Date', 'Status', 'State Type', 
        'Payment Type', 'Amount', 'Created By', 'Description', 'Additional Info', 'arrangement_id'
    ]].rename(columns={
        'arrangement_id': 'Arrangement ID'
    })
    
    # Create Failed Payments DataFrame
    df_failed = pd.DataFrame()
    if failed_payments_data:
        df_failed_display = pd.DataFrame(failed_payments_data)
        df_failed_display['Payment Date'] = pd.to_datetime(df_failed_display['payment_date']).dt.strftime('%Y-%m-%d')
        df_failed_display['Failure Reason'] = df_failed_display['failure_reason'].apply(lambda x: f"🔴 {x}")
        # Use helper function for days calculation
        df_failed_display['Days Ago'] = df_failed_display['payment_date'].apply(calculate_days_ago)
        df_failed = df_failed_display[['Payment Date', 'Failure Reason', 'Days Ago']].sort_values('Payment Date', ascending=False)
    
    # Build comprehensive summary with all payment information
    summary = f"""
# 💰 **Comprehensive Payment Analysis - Client {user_id}**

## 🎯 **Key Payment Indicators**
"""
    
    # Add last payment information
    # Helper function to calculate days ago with timezone handling
    def calculate_days_ago(date_str):
        try:
            date_obj = pd.to_datetime(date_str)
            # If timezone-aware, convert to naive datetime
            if date_obj.tz is not None:
                date_obj = date_obj.tz_localize(None)
            return (datetime.datetime.now() - date_obj).days
        except:
            return "Unknown"
    
    if last_successful_payment:
        days_since_success = calculate_days_ago(last_successful_payment['payment_date'])
        summary += f"""
### 🟢 **Last Successful Payment**
> 💵 **Amount:** `R {float(last_successful_payment['payment_amount']):,.2f}` | 📅 **Date:** `{last_successful_payment['payment_date']}` | ⏰ **Days Ago:** `{days_since_success}`
"""
    else:
        summary += """
### 🟢 **Last Successful Payment**
> ❌ **No successful payments found**
"""
    
    if last_valid_payment:
        days_since_valid = calculate_days_ago(last_valid_payment['payment_date'])
        summary += f"""
### ✅ **Last Valid Payment**
> 💵 **Amount:** `R {float(last_valid_payment['payment_amount']):,.2f}` | 📅 **Date:** `{last_valid_payment['payment_date']}` | ⏰ **Days Ago:** `{days_since_valid}`
"""
    else:
        summary += """
### ✅ **Last Valid Payment**
> ❌ **No valid payments found**
"""
    
    if last_reversed_payment:
        try:
            reversed_date_obj = pd.to_datetime(last_reversed_payment['date_received'])
            reversed_date = reversed_date_obj.strftime('%Y-%m-%d')
            days_since_reversed = calculate_days_ago(last_reversed_payment['date_received'])
        except:
            reversed_date = "Unknown"
            days_since_reversed = "Unknown"
        
        summary += f"""
### 🔄 **Last Reversed Payment**
> 💸 **Amount:** `R {float(last_reversed_payment['payment_amt']):,.2f}` | 📅 **Date:** `{reversed_date}` | ⏰ **Days Ago:** `{days_since_reversed}`
> 🔍 **Method:** `{last_reversed_payment['pay_method']}`
"""
    else:
        summary += """
### 🔄 **Last Reversed Payment**
> ✅ **No reversed payments found**
"""
    
    summary += "\n---\n"
    # Calculate summary statistics for arrangements
    total_arrangements = len(df)
    failed_count = len(df[df['arrangement_state'] == 'Failed'])
    successful_count = len(df[df['arrangement_state'] == 'Successful'])
    pending_count = len(df[df['arrangement_state'] == 'Pending'])
    total_amount = df['amount'].astype(float).sum()
    failed_amount = df[df['arrangement_state'] == 'Failed']['amount'].astype(float).sum()
    successful_amount = df[df['arrangement_state'] == 'Successful']['amount'].astype(float).sum()
    
    # Calculate failed payments statistics
    failed_payments_count = len(failed_payments_data) if failed_payments_data else 0
    
    # Get date range
    min_date = pd.to_datetime(df['create_ts']).min().strftime('%Y-%m-%d')
    max_date = pd.to_datetime(df['create_ts']).max().strftime('%Y-%m-%d')
    
    # Calculate success rate
    success_rate = (successful_count / total_arrangements * 100) if total_arrangements > 0 else 0
    
    # Add arrangement summary to existing summary
    summary += f"""
## 📊 **Payment Arrangements Overview**
> 📈 **Total Arrangements:** `{total_arrangements}` | 🎯 **Success Rate:** `{success_rate:.1f}%` | 💵 **Total Value:** `R {total_amount:,.2f}`

| 🔴 **Failed** | 🟢 **Successful** | 🟡 **Pending** | 💰 **Failed Amount** | ✅ **Successful Amount** |
|:---:|:---:|:---:|:---:|:---:|
| `{failed_count}` | `{successful_count}` | `{pending_count}` | `R {failed_amount:,.2f}` | `R {successful_amount:,.2f}` |

## 🚫 **Failed Payments Summary**
> 📊 **Total Failed Payments:** `{failed_payments_count}` payments failed | 📅 **Period:** `{min_date}` to `{max_date}`

## 📅 **Report Information**
> 📅 **Generated:** `{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}` | 🔄 **Data Coverage:** Arrangements + Individual Payments

---
"""
    
    # Add enhanced alerts based on all payment data
    alerts = []
    
    # Arrangement-based alerts
    if failed_count > successful_count:
        alerts.append("⚠️ **High arrangement failure rate** - More failed than successful arrangements")
    if failed_amount > 1000:
        alerts.append(f"💸 **Significant failed amount** - R {failed_amount:,.2f} in failed arrangements")
    if success_rate < 50:
        alerts.append(f"📉 **Low success rate** - Only {success_rate:.1f}% arrangement success rate")
    if pending_count > 0:
        alerts.append(f"⏳ **Pending arrangements** - {pending_count} arrangements awaiting resolution")
    
    # Payment-specific alerts
    if failed_payments_count > 5:
        alerts.append(f"🚫 **High payment failure count** - {failed_payments_count} individual payment failures")
    
    if last_successful_payment:
        try:
            days_since_success = calculate_days_ago(last_successful_payment['payment_date'])
            if isinstance(days_since_success, int):
                if days_since_success > 365:
                    alerts.append(f"📅 **No recent successful payments** - Last success was {days_since_success} days ago")
                elif days_since_success > 180:
                    alerts.append(f"⏰ **Long time since success** - Last successful payment was {days_since_success} days ago")
        except:
            pass
    else:
        alerts.append("❌ **No successful payments found** - Client has never made a successful payment")
    
    if last_reversed_payment:
        alerts.append("🔄 **Recent payment reversal** - Client has reversed payment history")
    
    # Payment pattern analysis
    if failed_payments_data:
        try:
            recent_failures = []
            for p in failed_payments_data:
                days_ago = calculate_days_ago(p['payment_date'])
                if isinstance(days_ago, int) and days_ago <= 90:
                    recent_failures.append(p)
            
            if len(recent_failures) >= 3:
                alerts.append(f"🔴 **Recent failure pattern** - {len(recent_failures)} failures in last 90 days")
        except:
            pass
    
    if alerts:
        summary += "### 🚨 **Critical Alerts & Risk Indicators**\n"
        for alert in alerts:
            summary += f"> {alert}\n"
        summary += "\n---\n"
    else:
        summary += "> ✅ **No critical alerts** - Payment profile appears healthy\n\n---\n"
    
    return df_final, df_failed, summary

def create_payment_history_app():
    """Create Gradio app for payment history display"""
    with gr.Blocks(title="💰 Payment History Dashboard") as app:
        gr.Markdown("# 💰 **Payment History Dashboard**")
        gr.Markdown("### 📋 Complete payment arrangements and transaction history")
        
        with gr.Row():
            with gr.Column(scale=2):
                user_id_input = gr.Textbox(
                    label="🆔 Client ID", 
                    placeholder="Enter client ID...",
                    value="963262"  # Using the example user_id from your data
                )
            with gr.Column(scale=1):
                search_button = gr.Button("🚀 Search History", variant="primary", size="lg")
        
        # Summary section
        summary_output = gr.Markdown()
        
        # Main arrangements table
        gr.Markdown("### 📊 **Payment Arrangements History**")
        dataframe_output = gr.Dataframe(
            headers=[
                "Created Date", "Created Time", "Payment Date", "Status", "State Type", 
                "Payment Type", "Amount", "Created By", "Description", "Additional Info", "Arrangement ID"
            ],
            max_height=400,
            show_search='filter',
            wrap=False,
            interactive=False
        )
        
        # Failed payments table
        gr.Markdown("### 🚫 **Individual Failed Payments**")
        failed_payments_output = gr.Dataframe(
            headers=["Payment Date", "Failure Reason", "Days Ago"],
            max_height=300,
            show_search='filter',
            wrap=False,
            interactive=False
        )
        
        def update_payment_history(user_id):
            df_arrangements, df_failed, summary = display_payment_history(user_id)
            return summary, df_arrangements, df_failed
        
        # Event handlers
        search_button.click(
            fn=update_payment_history, 
            inputs=user_id_input, 
            outputs=[summary_output, dataframe_output, failed_payments_output]
        )
        
        user_id_input.submit(
            fn=update_payment_history, 
            inputs=user_id_input, 
            outputs=[summary_output, dataframe_output, failed_payments_output]
        )
        
        # Load initial data
        initial_df_arrangements, initial_df_failed, initial_summary = display_payment_history("963262")
        summary_output.value = initial_summary
        dataframe_output.value = initial_df_arrangements
        failed_payments_output.value = initial_df_failed
    
    return app

# Alternative function for integration into existing dashboard
def get_payment_history_summary(user_id):
    """Get a concise payment history summary for dashboard integration"""
    data = CartrackSQLDatabase.get_client_payment_history.invoke(user_id)
    if not data:
        return "❌ No payment history available"
    
    df = pd.DataFrame(data)
    total = len(df)
    failed = len(df[df['arrangement_state'] == 'Failed'])
    successful = len(df[df['arrangement_state'] == 'Successful'])
    success_rate = (successful / total * 100) if total > 0 else 0
    total_amount = df['amount'].astype(float).sum()
    
    return f"""
### 💰 **Payment History Summary**
> 📊 **{total} arrangements** | 🎯 **{success_rate:.1f}% success** | 💵 **R {total_amount:,.2f} total**
> 🟢 **{successful} successful** | 🔴 **{failed} failed**
"""

# Usage example - you can integrate this into your main app
if __name__ == "__main__":
    demo = create_payment_history_app()
    demo.launch()