import gradio as gr
import pandas as pd
import sys
import datetime
sys.path.append("../..")
from src.Database import CartrackSQLDatabase

def display_client_banking_details(user_id=None):
    """Display client banking details information using markdown"""
    if not user_id:
        return "Please enter a client ID and click Search to view banking details."
    
    # Get banking details data
    data = CartrackSQLDatabase.get_client_banking_details.invoke(user_id)[0]
    
    if not data:
        return f"❌ **Banking details not found**\n\nNo banking information available for client ID: {user_id}"
    
    # Helper function to format values with fallback
    def format_value(value, fallback="N/A"):
        return str(value) if value is not None else fallback
    
    # Helper function to get status emoji and text
    def get_account_status():
        if data.get('rejected') == 1:
            return "🔴", "Rejected"
        elif data.get('isactive') == 1:
            return "🟢", "Active"
        else:
            return "🟡", "Inactive"
    
    def get_billing_status():
        if data.get('suspend_billing'):
            return "⏸️", "Suspended"
        else:
            return "▶️", "Active"
    
    # Get status indicators
    account_emoji, account_status = get_account_status()
    billing_emoji, billing_status = get_billing_status()
    
    # Format credit card information
    credit_card_info = "💳 *No credit card on file*"
    if data.get('credit_card_number'):
        credit_card_info = f"💳 **Card:** `{format_value(data.get('credit_card_number'))}`<br>📅 **Expires:** `{format_value(data.get('exp_date'))}`<br>🔒 **CVV:** `{'***' if data.get('cdv') else 'N/A'}`"
    
    # Format mandate information
    mandate_status = "✅ Active" if data.get('mandate') else "❌ Not Active"
    primary_status = "⭐ Primary" if not data.get('non_primary_account') else "📄 Secondary"
    
    # Format rejected reason
    rejected_info = ""
    if data.get('rejected') == 1 and data.get('rejected_reason'):
        rejected_info = f"""
### ⚠️ **ACCOUNT REJECTION DETAILS**
> 🚫 **Reason:** `{data.get('rejected_reason')}`
> 📅 **Status:** Account has been rejected and requires attention

---
"""
    
    # Build comprehensive markdown output
    markdown_output = f"""
# 🏦 **BANKING DETAILS - {format_value(data.get('first_name'))} {format_value(data.get('last_name'))}**

> 🆔 **User ID:** `{format_value(data.get('user_id'))}` | 🏦 **Account ID:** `{format_value(data.get('user_bank_account_id'))}` | {account_emoji} **Status:** `{account_status}`

{rejected_info}

## 💰 **ACCOUNT INFORMATION**

| 🏦 **Bank Details** | 👤 **Account Holder** | 💳 **Payment Method** | ⚙️ **Settings** |
|:---|:---|:---|:---|
| 🏛️ **Bank:** `{format_value(data.get('bank_name'))}`<br><br>🏢 **Branch:** `{format_value(data.get('branch_name'))}`<br><br>🔢 **Branch Code:** `{format_value(data.get('branch_code'))}`<br><br>💼 **Account Type:** `{format_value(data.get('bank_account_type'))}`<br><br>📂 **Category:** `{format_value(data.get('bank_account_category'))}` | 👤 **Name:** `{format_value(data.get('account_name'))}`<br><br>🆔 **ID Number:** `{format_value(data.get('id_number'))}`<br><br>🛂 **Passport:** `{format_value(data.get('passport_number'))}`<br><br>🔒 **Account #:** `{format_value(data.get('account_number'))}`{2*'<br>'} | 💵 **Method:** `{format_value(data.get('payment_method'))}`<br><br>📋 **Invoice Option:** `{format_value(data.get('invoice_option'))}`<br><br>📅 **Debit Date:** `{format_value(data.get('debit_date'))}`<br><br>📊 **Terms:** `{format_value(data.get('payment_terms'))} days`<br><br>⚡ **Service:** `{format_value(data.get('pacs_service'))}` | 📝 **Mandate:** `{mandate_status}`<br><br>⭐ **Priority:** `{primary_status}`<br><br>{billing_emoji} **Billing:** `{billing_status}`<br><br>💰 **Debit Day:** `{format_value(data.get('debit_run_day_display_name'))}`<br><br>🏦 **State:** `{format_value(data.get('account_state'))}` |

---

## 🔐 **AUTHORIZATION & PERMISSIONS**

| ✅ **Permissions** | 🏦 **Debit Settings** | 👨‍💼 **Management** |
|:---|:---|:---|
| 🔄 **Allow Debit:** `{'✅ Yes' if data.get('allow_debit') else '❌ No'}`<br><br>📝 **Allow Mandate:** `{'✅ Yes' if data.get('allow_mandate') else '❌ No'}`<br><br>💰 **Savings Debit:** `{'✅ Yes' if data.get('allow_debit_savings_account') else '❌ No'}`<br><br>🔍 **AVS Check:** `{'✅ Enabled' if data.get('avs') else '❌ Disabled'}` | 📅 **Salary Date:** `{format_value(data.get('salary_day'))}`<br><br>💼 **Debit on Salary:** `{'✅ Yes' if data.get('debit_on_salary_date') else '❌ No'}`<br><br>🔄 **Separate Billing:** `{'✅ Yes' if data.get('separate_billing_run') else '❌ No'}`<br><br>⚖️ **Pre-Legal:** `{'⚠️ Yes' if data.get('pre_legal') else '✅ No'}` | 👨‍💼 **Credit Controller:** `{format_value(data.get('credit_controller'))}`<br><br>🏷️ **Classification:** `{format_value(data.get('client_classification')) or 'Standard'}`<br><br>🔒 **Disabled:** `{'❌ Yes' if data.get('disabled') else '✅ No'}`{2*'<br>'} |

---

## 💳 **CREDIT CARD INFORMATION**

{credit_card_info}

---

### 📋 **Banking Report Summary**
> 📅 **Generated:** `{datetime.datetime.now().strftime("%B %d, %Y")}` | ⏰ **Time:** `{datetime.datetime.now().strftime("%I:%M:%S %p")}` | 🏦 **Bank ID:** `{format_value(data.get('bank_id'))}`

### 🚨 **Key Alerts**
"""
    
    # Add alerts based on data conditions
    alerts = []
    if data.get('rejected') == 1:
        alerts.append("🚫 **Account is REJECTED** - Immediate attention required")
    if data.get('suspend_billing'):
        alerts.append("⏸️ **Billing is SUSPENDED** - No charges will be processed")
    if not data.get('allow_debit'):
        alerts.append("❌ **Debit orders DISABLED** - Manual payments required")
    if data.get('pre_legal'):
        alerts.append("⚖️ **Pre-legal status** - Account under review")
    if data.get('disabled'):
        alerts.append("🔒 **Account DISABLED** - Contact required")
    
    if alerts:
        for alert in alerts:
            markdown_output += f"\n> {alert}"
    else:
        markdown_output += "\n> ✅ **No critical alerts** - Account appears to be in good standing"
    
    return markdown_output

def create_banking_details_app():
    """Create Gradio app for banking details display"""
    with gr.Blocks(title="🏦 Banking Details Dashboard") as app:
        gr.Markdown("# 💰 **CLIENT BANKING DETAILS DASHBOARD**")
        gr.Markdown("### 🏦 Comprehensive banking and payment information")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 🔎 **Search Panel**")
                user_id_input = gr.Textbox(
                    label="🆔 Client ID", 
                    placeholder="Enter client ID...",
                    value="28173"
                )
                search_button = gr.Button("🚀 Search Banking Details", variant="primary", size="lg")
            
            with gr.Column(scale=4):
                gr.Markdown("#### 🏦 **Banking Information**")
                banking_markdown = gr.Markdown(
                    elem_id="banking-output"
                )
        
        def update_banking_details(user_id):
            return display_client_banking_details(user_id)
        
        # Event handlers
        search_button.click(
            fn=update_banking_details, 
            inputs=user_id_input, 
            outputs=banking_markdown
        )
        
        user_id_input.submit(
            fn=update_banking_details, 
            inputs=user_id_input, 
            outputs=banking_markdown
        )
        
        # Load initial data
        initial_markdown = display_client_banking_details("28173")
        banking_markdown.value = initial_markdown
    
    return app

# Usage example - you can integrate this into your main app
if __name__ == "__main__":
    demo = create_banking_details_app()
    demo.launch()