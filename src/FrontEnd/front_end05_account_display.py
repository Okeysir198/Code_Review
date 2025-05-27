import gradio as gr
import pandas as pd
import sys
import datetime
sys.path.append("../..")
from src.Database import CartrackSQLDatabase

def display_client_account_overview(user_id=None):
    """Display client account overview using multiple DataFrames for better organization"""
    if not user_id:
        empty_df = pd.DataFrame(columns=['Field', 'Value'])
        return empty_df, empty_df, empty_df, empty_df, empty_df, "Please enter a client ID and click Search to view account overview.", "Please enter a client ID."
    
    # Fetch data from multiple sources
    data = CartrackSQLDatabase.get_client_account_overview.invoke(user_id)
    account_status_raw = CartrackSQLDatabase.get_client_account_status.invoke(user_id)
    billing_analysis_raw = CartrackSQLDatabase.get_client_billing_analysis.invoke(user_id)
    
    if not data:
        empty_df = pd.DataFrame(columns=['Field', 'Value'])
        return empty_df, empty_df, empty_df, empty_df, empty_df, f"❌ **No data found for client ID: {user_id}**", "❌ No data found"
    
    # Combine all data sources for main overview
    combined_data = data.copy()
    if account_status_raw:
        combined_data.update(account_status_raw[0])
    if billing_analysis_raw:
        combined_data.update(billing_analysis_raw[0])
    
    # 1. CLIENT INFORMATION DataFrame
    client_info = [
        ("👤 Client Name", combined_data.get('client', 'N/A')),
        ("🆔 User ID", combined_data.get('user_id', 'N/A')),
        ("👥 Client Type", combined_data.get('client_type', 'N/A')),
        ("📅 Introduction Date", combined_data.get('intro_date', 'N/A')),
        ("⭐ Rating", f"{get_stars(combined_data.get('rating', '0'))} ({combined_data.get('rating', '0')})"),
        ("👨‍💼 Sales Agent", combined_data.get('sale_agent', 'N/A')),
        ("👩‍💼 Responsible Agent", combined_data.get('responsible_agent', 'N/A')),
        ("🏢 Department", combined_data.get('department', 'N/A')),
    ]
    df_client = pd.DataFrame(client_info, columns=['Field', 'Value'])
    
    # 2. ACCOUNT STATUS DataFrame
    account_status_info = [
        ("📊 Account Status", f"{get_status_emoji(combined_data.get('account_status', 'Unknown'))} {combined_data.get('account_status', 'Unknown')}"),
        ("🔧 Service Status", f"{get_status_emoji(combined_data.get('service_status', 'Unknown'))} {combined_data.get('service_status', 'Unknown')}"),
        ("💳 Payment Status", f"{get_status_emoji(combined_data.get('payment_status', 'Unknown'))} {combined_data.get('payment_status', 'Unknown')}"),
        ("🚗 Active/Total Vehicles", f"{combined_data.get('vehicle_count_active', '0')}/{combined_data.get('vehicle_count_total', '0')}"),
        ("📞 Contactable", get_yes_no(combined_data.get('contactable', ''))),
        ("⚠️ Disputes", get_yes_no(combined_data.get('disputes', ''), reverse=True)),
        ("🔒 Blacklisted", get_yes_no(combined_data.get('blacklisted', ''), reverse=True)),
        ("📞 Last Agent Contact", combined_data.get('last_agent_contact_date', 'N/A')),
    ]
    df_status = pd.DataFrame(account_status_info, columns=['Field', 'Value'])
    
    # 3. FINANCIAL INFORMATION DataFrame
    total_invoices = float(str(combined_data.get('total_invoices', '0')).replace(',', '').replace('R', '').strip() or '0')
    total_paid = float(str(combined_data.get('total_paid', '0')).replace(',', '').replace('R', '').strip() or '0')
    payment_rate = (total_paid / total_invoices * 100) if total_invoices > 0 else 0
    balance = total_invoices - total_paid
    
    financial_info = [
        ("💰 Total Invoiced", f"R {total_invoices:,.2f}"),
        ("💵 Total Paid", f"R {total_paid:,.2f}"),
        ("📊 Payment Rate", f"{payment_rate:.1f}% {get_percentage_emoji(payment_rate)}"),
        ("⚖️ Current Balance", f"R {balance:,.2f} {'🔴' if balance > 0 else '🟢' if balance < 0 else '⚪'}"),
        ("📅 Last Payment", combined_data.get('last_successful_payment_date', 'Never')),
        ("💳 Monthly Debit Order", f"R {format_currency(combined_data.get('monthly_debit_order_total', '0'))}"),
        ("📈 Annual Revenue", f"R {format_currency(combined_data.get('annual_recurring_revenue_excl_vat', '0'))}"),
        ("📊 Monthly CRV", f"R {format_currency(combined_data.get('crv', '0'))}"),
    ]
    df_financial = pd.DataFrame(financial_info, columns=['Field', 'Value'])
    
    # 4. BILLING DETAILS DataFrame
    billing_info = [
        ("📅 Next Invoice Date", combined_data.get('next_invoice_date', 'N/A')),
        ("📅 Last Invoice Date", combined_data.get('last_invoice_date', 'N/A')),
        ("🏦 Debit Order Date", combined_data.get('debit_order_date', 'N/A')),
        ("📋 Payment Arrangement", combined_data.get('payment_arrangement', 'N/A')),
        ("💳 Invoice Option", combined_data.get('invoice_option', 'N/A')),
        ("👨‍💼 Credit Controller", combined_data.get('credit_controller', 'N/A')),
        ("🏢 Branch", combined_data.get('branch_name', 'N/A')),
        ("⚖️ Pre-Legal Billing", combined_data.get('pre_legal_billing', 'N/A')),
    ]
    df_billing = pd.DataFrame(billing_info, columns=['Field', 'Value'])
    
    # 5. BILLING ANALYSIS DataFrame (from get_client_billing_analysis)
    df_billing_analysis = pd.DataFrame()
    if billing_analysis_raw and len(billing_analysis_raw) > 0:
        billing_data = billing_analysis_raw[0]
        billing_analysis_details = []
        for key, value in billing_data.items():
            if value not in [None, '', 'N/A']:
                formatted_key = format_field_name(key)
                formatted_value = format_field_value_with_status(key, value)
                billing_analysis_details.append((formatted_key, formatted_value))
        df_billing_analysis = pd.DataFrame(billing_analysis_details, columns=['Field', 'Value'])
    
    # 6. DETAILED ACCOUNT STATUS as Markdown (from get_client_account_status)
    account_status_markdown = "### 🔍 **Detailed Account Status**\n\n"
    if account_status_raw and len(account_status_raw) > 0:
        account_status_data = account_status_raw[0]
        account_status_markdown += "| **Field** | **Value** |\n|:---|:---|\n"
        for key, value in account_status_data.items():
            if value not in [None, '', 'N/A']:
                formatted_key = format_field_name(key).replace('📋 ', '').replace('📊 ', '').replace('💰 ', '').replace('💵 ', '').replace('📅 ', '').replace('⚠️ ', '').replace('⏰ ', '').replace('👨‍💼 ', '').replace('📈 ', '').replace('🚗 ', '').replace('🏢 ', '').replace('🏭 ', '').replace('❌ ', '').replace('📝 ', '').replace('💳 ', '').replace('⚖️ ', '')
                formatted_value = format_field_value_with_status(key, value)
                account_status_markdown += f"| **{formatted_key}** | {formatted_value} |\n"
    else:
        account_status_markdown += "> 🚫 *No detailed account status available*\n"
    
    # Calculate key metrics for summary
    risk_level = get_risk_level(payment_rate, balance, combined_data.get('account_status', ''))
    vehicle_health = get_vehicle_health(combined_data.get('vehicle_count_active', '0'), combined_data.get('vehicle_count_total', '0'))
    
    # Create summary
    summary = f"""
# 🏢 **{combined_data.get('client', 'Unknown Client')}**

📊 **Key Metrics:** {payment_rate:.1f}% Payment Rate • R {balance:,.2f} Balance • {combined_data.get('vehicle_count_active', '0')}/{combined_data.get('vehicle_count_total', '0')} Vehicles Active  
⚡ **Status:** {get_status_emoji(combined_data.get('account_status', 'Unknown'))} {combined_data.get('account_status', 'Unknown')} • {risk_level} • {vehicle_health}  
📅 **Generated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}

{get_alerts(combined_data, balance, payment_rate)}
"""
    
    return df_client, df_status, df_financial, df_billing, df_billing_analysis, summary, account_status_markdown

def get_status_emoji(status):
    """Get emoji for status"""
    if not status:
        return "❓"
    s = str(status).lower()
    if any(word in s for word in ['active', 'paid', 'good']):
        return "✅"
    elif any(word in s for word in ['arrears', 'overdue', 'suspended', 'locked']):
        return "🔴"
    elif any(word in s for word in ['pending', 'caution']):
        return "⚠️"
    else:
        return "ℹ️"

def get_yes_no(value, reverse=False):
    """Format yes/no values"""
    if not value:
        return "❓ Unknown"
    v = str(value).lower()
    if v in ['yes', 'true', '1']:
        return "❌ Yes" if reverse else "✅ Yes"
    elif v in ['no', 'false', '0']:
        return "✅ No" if reverse else "❌ No"
    else:
        return f"ℹ️ {value}"

def get_percentage_emoji(percentage):
    """Get emoji for percentage"""
    if percentage >= 90:
        return "🟢"
    elif percentage >= 70:
        return "🟡"
    elif percentage >= 50:
        return "🟠"
    else:
        return "🔴"

def get_stars(rating):
    """Get star rating"""
    try:
        return "⭐" * min(int(float(str(rating))), 5)
    except:
        return "☆"

def format_currency(value):
    """Format currency"""
    try:
        return f"{float(str(value).replace(',', '').replace('R', '').strip()):,.2f}"
    except:
        return str(value)

def get_risk_level(payment_rate, balance, account_status):
    """Calculate risk level"""
    risk_score = 0
    
    # Payment rate factor
    if payment_rate < 50:
        risk_score += 3
    elif payment_rate < 70:
        risk_score += 2
    elif payment_rate < 90:
        risk_score += 1
    
    # Balance factor
    if balance > 5000:
        risk_score += 2
    elif balance > 1000:
        risk_score += 1
    
    # Status factor
    if 'arrears' in str(account_status).lower():
        risk_score += 2
    
    if risk_score <= 1:
        return "🟢 Low Risk"
    elif risk_score <= 3:
        return "🟡 Medium Risk"
    elif risk_score <= 5:
        return "🟠 High Risk"
    else:
        return "🔴 Critical Risk"

def get_vehicle_health(active, total):
    """Get vehicle health status"""
    try:
        a, t = int(str(active)), int(str(total))
        if t == 0:
            return "❓ No Vehicles"
        ratio = a / t
        if ratio == 1:
            return "🟢 All Active"
        elif ratio >= 0.8:
            return "🟡 Most Active"
        elif ratio >= 0.5:
            return "🟠 Some Active"
        else:
            return "🔴 Few Active"
    except:
        return "❓ Unknown"

def format_field_name(field_name):
    """Format field names for display"""
    field_name = str(field_name).replace('_', ' ').title()
    
    # Add appropriate emojis
    emoji_map = {
        'Account State': '📊 Account State',
        'Balance Total': '💰 Balance Total',
        'Balance Current': '💵 Balance Current',
        'Balance 30 Days': '📅 Balance 30 Days',
        'Balance 60 Days': '📅 Balance 60 Days', 
        'Balance 90 Days': '📅 Balance 90 Days',
        'Balance 120 Days': '⚠️ Balance 120+ Days',
        'Age Days': '⏰ Age Days',
        'Credit Controller': '👨‍💼 Credit Controller',
        'Annual Recurring Revenue Excl Vat': '📈 Annual Revenue',
        'Crv': '📊 Monthly CRV',
        'Vehicle Count': '🚗 Vehicle Count',
        'Branch Name': '🏢 Branch Name',
        'First Contract Start Date': '📅 First Contract Date',
        'Industry Sector': '🏭 Industry Sector',
        'Rejected': '❌ Rejected Status',
        'Rejected Reason': '📝 Rejection Reason',
        'Payment Arrangement': '📋 Payment Arrangement',
        'Invoice Option': '💳 Invoice Option',
        'Pre Legal Billing': '⚖️ Pre-Legal Billing'
    }
    
    return emoji_map.get(field_name, f"📋 {field_name}")

def format_field_value_with_status(field_name, value):
    """Format field values with appropriate status indicators"""
    if not value or value in ['', 'None', 'null']:
        return "N/A"
    
    field_lower = str(field_name).lower()
    value_str = str(value)
    
    # Currency fields
    if any(word in field_lower for word in ['balance', 'revenue', 'crv', 'amount']):
        try:
            num_value = float(value_str.replace(',', '').replace('R', '').strip())
            emoji = '🔴' if num_value > 0 and 'balance' in field_lower else '🟢' if num_value < 0 else '⚪'
            return f"R {num_value:,.2f} {emoji}"
        except:
            return value_str
    
    # Status fields
    elif any(word in field_lower for word in ['status', 'state', 'rejected']):
        return f"{get_status_emoji(value_str)} {value_str}"
    
    # Date fields
    elif any(word in field_lower for word in ['date', 'created', 'updated']):
        if 'T' in value_str:
            return f"📅 {value_str[:10]}"
        return f"📅 {value_str}"
    
    # Age/days fields
    elif 'age' in field_lower or 'days' in field_lower:
        try:
            days = int(float(value_str))
            emoji = '🟢' if days <= 30 else '🟡' if days <= 60 else '🟠' if days <= 90 else '🔴'
            return f"{days} days {emoji}"
        except:
            return value_str
    
    # Vehicle count
    elif 'vehicle' in field_lower and 'count' in field_lower:
        return f"🚗 {value_str}"
    
    # Default formatting
    else:
        return value_str

def get_alerts(data, balance, payment_rate):
    """Generate critical alerts"""
    alerts = []
    
    if balance > 5000:
        alerts.append(f"🚨 **HIGH BALANCE:** R {balance:,.2f} outstanding")
    
    if payment_rate < 50:
        alerts.append(f"⚠️ **LOW PAYMENT RATE:** {payment_rate:.1f}%")
    
    if 'arrears' in str(data.get('account_status', '')).lower():
        alerts.append("🔴 **ACCOUNT IN ARREARS**")
    
    if str(data.get('blacklisted', '')).lower() in ['yes', 'true', '1']:
        alerts.append("❌ **BLACKLISTED CLIENT**")
    
    if alerts:
        return "\n> " + "\n> ".join(alerts) + "\n"
    return ""

def create_client_overview_app():
    with gr.Blocks(title="🏢 Client Account Overview") as app:
        gr.Markdown("# 📊 **CLIENT ACCOUNT OVERVIEW**")
        
        with gr.Row():
            user_id_input = gr.Textbox(label="🆔 Client ID", value="28173", scale=2)
            search_button = gr.Button("🚀 Search", variant="primary", scale=1)
        
        # Summary section
        summary_output = gr.Markdown()
        
        # Detailed Account Status as Markdown (moved up)
        account_status_markdown_output = gr.Markdown()

        # Main content layout
        with gr.Row():
            # Left column - All tables except Billing Analysis
            with gr.Column(scale=1):
                gr.Markdown("### 👤 **Client Information**")
                df_client_output = gr.Dataframe(
                    headers=["Field", "Value"], 
                    wrap=False,
                    column_widths=["40%", "60%"],
                    interactive=False
                )
                
                gr.Markdown("### 📊 **Account Status**")
                df_status_output = gr.Dataframe(
                    headers=["Field", "Value"],
                    wrap=False,
                    column_widths=["40%", "60%"],
                    interactive=False
                )
                
                gr.Markdown("### 💰 **Financial Information**")
                df_financial_output = gr.Dataframe(
                    headers=["Field", "Value"],
                    wrap=False,
                    column_widths=["40%", "60%"],
                    interactive=False
                )
                
                gr.Markdown("### 💳 **Billing Details**")
                df_billing_output = gr.Dataframe(
                    headers=["Field", "Value"],
                    wrap=False,
                    column_widths=["40%", "60%"],
                    interactive=False
                )
            
            # Right column - Billing Analysis only
            with gr.Column(scale=1):
                gr.Markdown("### 📈 **Billing Analysis**")
                df_billing_analysis_output = gr.Dataframe(
                    headers=["Field", "Value"],
                    wrap=False,
                    column_widths=["40%", "60%"],
                    interactive=False,
                    max_height=1800,
                )
        
        def update_display(user_id):
            df_client, df_status, df_financial, df_billing, df_billing_analysis, summary, account_status_markdown = display_client_account_overview(user_id)
            return summary, account_status_markdown, df_client, df_status, df_financial, df_billing, df_billing_analysis
        
        search_button.click(
            update_display, 
            user_id_input, 
            [summary_output, account_status_markdown_output, df_client_output, df_status_output, 
             df_financial_output, df_billing_output, df_billing_analysis_output]
        )
        user_id_input.submit(
            update_display, 
            user_id_input, 
            [summary_output, account_status_markdown_output, df_client_output, df_status_output, 
             df_financial_output, df_billing_output, df_billing_analysis_output]
        )
        
        # Load initial data
        initial_client, initial_status, initial_financial, initial_billing, initial_billing_analysis, initial_summary, initial_account_status_markdown = display_client_account_overview("28173")
        summary_output.value = initial_summary
        account_status_markdown_output.value = initial_account_status_markdown
        df_client_output.value = initial_client
        df_status_output.value = initial_status
        df_financial_output.value = initial_financial
        df_billing_output.value = initial_billing
        df_billing_analysis_output.value = initial_billing_analysis
    
    return app

# if __name__ == "__main__":
#     demo = create_client_overview_app()
#     demo.launch()