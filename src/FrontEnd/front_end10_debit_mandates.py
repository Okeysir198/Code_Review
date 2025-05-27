import gradio as gr
import pandas as pd
import sys
import datetime
sys.path.append("../..")
from src.Database import CartrackSQLDatabase

def display_client_debit_mandates(user_id=None):
    """Display comprehensive client debit mandates using DataFrame + summary"""
    if not user_id:
        return None, "Please enter a client ID and click Search to view debit mandates."
    
    # Helper function to calculate days ago with timezone handling
    def calculate_days_ago(date_str):
        try:
            date_obj = pd.to_datetime(date_str)
            if date_obj.tz is not None:
                date_obj = date_obj.tz_localize(None)
            return (datetime.datetime.now() - date_obj).days
        except:
            return "Unknown"
    
    # Get debit mandates data
    data = CartrackSQLDatabase.get_client_debit_mandates.invoke(user_id)
    if not data:
        return None, f"❌ **No debit mandates found for client ID: {user_id}**"
    
    df = pd.DataFrame(data)
    if df.empty:
        return None, f"❌ **No mandate records found for client ID: {user_id}**"
    
    # Process DataFrame for display
    df_display = df.copy()
    
    # Format dates
    df_display['Created Date'] = pd.to_datetime(df['create_ts']).dt.strftime('%Y-%m-%d')
    df_display['Created Time'] = pd.to_datetime(df['create_ts']).dt.strftime('%H:%M')
    df_display['First Collection'] = pd.to_datetime(df['first_collection_date']).dt.strftime('%Y-%m-%d')
    df_display['Initiation Date'] = pd.to_datetime(df['mandate_initiation_date']).dt.strftime('%Y-%m-%d')
    
    # Format mandate state with emojis
    state_map = {
        'Created': '🟡 Created',
        'Active': '🟢 Active',
        'Suspended': '🟠 Suspended',
        'Cancelled': '🔴 Cancelled',
        'Expired': '⚫ Expired',
        'Pending': '🔵 Pending'
    }
    df_display['State'] = df['debicheck_mandate_state'].map(state_map).fillna('❓ ' + df['debicheck_mandate_state'].astype(str))
    
    # Format service type
    service_map = {
        'TT1': '🏦 TT1 (Standard)',
        'PTP': '💰 PTP (Promise to Pay)',
        'service': '⚙️ Standard Service'
    }
    df_display['Service Type'] = df['service'].map(service_map).fillna('📋 ' + df['service'].astype(str))
    
    # Format frequency
    frequency_map = {
        'MNTH': '📅 Monthly',
        'WEEK': '📅 Weekly',
        'YEAR': '📅 Yearly',
        'OOFF': '⚡ One-off'
    }
    df_display['Frequency'] = df['frequency'].map(frequency_map).fillna('📅 ' + df['frequency'].astype(str))
    
    # Format amounts
    df_display['Collection Amount'] = df['collection_amount'].astype(float).apply(lambda x: f"R {x:,.2f}")
    df_display['Maximum Amount'] = df['maximum_collection_amount'].astype(float).apply(lambda x: f"R {x:,.2f}")
    df_display['First Amount'] = df['first_collection_amount'].astype(float).apply(lambda x: f"R {x:,.2f}")
    
    # Format account details
    df_display['Account Info'] = df.apply(lambda row: f"{row['debtor_account_name'][:20]}... | {row['debtor_account_number']}", axis=1)
    df_display['Branch Info'] = df.apply(lambda row: f"Branch: {row['debtor_branch_number']}", axis=1)
    
    # Format authentication status
    df_display['Auth Status'] = df['authenticated'].apply(lambda x: '✅ Authenticated' if x else '❌ Not Authenticated')
    
    # Calculate days until first collection
    df_display['Days to Collection'] = df['first_collection_date'].apply(
        lambda x: calculate_days_ago(x) * -1 if calculate_days_ago(x) != "Unknown" else "Unknown"
    )
    
    # Select columns for final display
    df_final = df_display[[
        'Created Date', 'Created Time', 'First Collection', 'State', 'Service Type', 
        'Frequency', 'Collection Amount', 'Maximum Amount', 'Auth Status', 
        'Account Info', 'Days to Collection', 'client_debit_order_mandate_id'
    ]].rename(columns={
        'client_debit_order_mandate_id': 'Mandate ID'
    })
    
    # Calculate summary statistics
    total_mandates = len(df)
    created_count = len(df[df['debicheck_mandate_state'] == 'Created'])
    active_count = len(df[df['debicheck_mandate_state'] == 'Active'])
    suspended_count = len(df[df['debicheck_mandate_state'] == 'Suspended'])
    cancelled_count = len(df[df['debicheck_mandate_state'] == 'Cancelled'])
    
    authenticated_count = len(df[df['authenticated'] == True])
    not_authenticated_count = total_mandates - authenticated_count
    
    total_collection_amount = df['collection_amount'].astype(float).sum()
    average_amount = df['collection_amount'].astype(float).mean()
    max_single_amount = df['collection_amount'].astype(float).max()
    
    # Get unique account details
    unique_accounts = df['debtor_account_number'].nunique()
    unique_branches = df['debtor_branch_number'].nunique()
    
    # Calculate date ranges
    earliest_creation = pd.to_datetime(df['create_ts']).min().strftime('%Y-%m-%d')
    latest_creation = pd.to_datetime(df['create_ts']).max().strftime('%Y-%m-%d')
    earliest_collection = pd.to_datetime(df['first_collection_date']).min().strftime('%Y-%m-%d')
    latest_collection = pd.to_datetime(df['first_collection_date']).max().strftime('%Y-%m-%d')
    
    # Service type breakdown
    service_counts = df['service'].value_counts().to_dict()
    
    # Build comprehensive summary
    summary = f"""
# 🏦 **Debit Mandates Dashboard - Client {user_id}**

## 📊 **Mandate Overview**
> 📈 **Total Mandates:** `{total_mandates}` | 💰 **Total Collection Value:** `R {total_collection_amount:,.2f}` | 📅 **Average Amount:** `R {average_amount:,.2f}`

| 🟡 **Created** | 🟢 **Active** | 🟠 **Suspended** | 🔴 **Cancelled** | ✅ **Authenticated** | ❌ **Not Auth** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| `{created_count}` | `{active_count}` | `{suspended_count}` | `{cancelled_count}` | `{authenticated_count}` | `{not_authenticated_count}` |

## 💳 **Account & Collection Details**
> 🏦 **Unique Accounts:** `{unique_accounts}` | 🏢 **Unique Branches:** `{unique_branches}` | 💰 **Highest Amount:** `R {max_single_amount:,.2f}`

## 📅 **Timeline Information**
> 🗓️ **Created Between:** `{earliest_creation}` - `{latest_creation}`
> 💰 **Collections Between:** `{earliest_collection}` - `{latest_collection}`

## ⚙️ **Service Breakdown**
"""
    
    # Add service type breakdown
    for service, count in service_counts.items():
        service_display = service_map.get(service, f"📋 {service}")
        summary += f"> {service_display}: `{count} mandates`\n"
    
    summary += f"""

## 📅 **Report Information**
> 📅 **Generated:** `{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}` | 🔄 **Data Source:** Debit Order Mandates

---
"""
    
    # Add intelligent alerts
    alerts = []
    
    # Authentication alerts
    auth_rate = (authenticated_count / total_mandates * 100) if total_mandates > 0 else 0
    if auth_rate < 50:
        alerts.append(f"🔐 **Low authentication rate** - Only {auth_rate:.1f}% of mandates are authenticated")
    
    # State-based alerts
    if cancelled_count > active_count:
        alerts.append("🚨 **High cancellation rate** - More cancelled than active mandates")
    
    if suspended_count > 0:
        alerts.append(f"⏸️ **Suspended mandates** - {suspended_count} mandates currently suspended")
    
    # Amount-based alerts
    if max_single_amount > 1000:
        alerts.append(f"💰 **High-value mandate** - Maximum collection amount is R {max_single_amount:,.2f}")
    
    # Check for upcoming collections (within 7 days)
    upcoming_collections = []
    for _, row in df.iterrows():
        days_to_collect = calculate_days_ago(row['first_collection_date']) * -1
        if isinstance(days_to_collect, int) and -7 <= days_to_collect <= 7:
            upcoming_collections.append(row)
    
    if upcoming_collections:
        alerts.append(f"📅 **Upcoming collections** - {len(upcoming_collections)} mandates due within 7 days")
    
    # Check for multiple mandates on same day
    collection_days = df['collection_day'].value_counts()
    busy_days = collection_days[collection_days > 1]
    if not busy_days.empty:
        max_day = busy_days.index[0]
        max_count = busy_days.iloc[0]
        alerts.append(f"📆 **Collection clustering** - {max_count} mandates scheduled for day {max_day}")
    
    # Check for recent mandate creation (last 7 days)
    recent_mandates = [m for _, m in df.iterrows() if calculate_days_ago(m['create_ts']) <= 7]
    if len(recent_mandates) >= 3:
        alerts.append(f"🆕 **Recent mandate activity** - {len(recent_mandates)} mandates created in last 7 days")
    
    if alerts:
        summary += "### 🚨 **Key Alerts & Insights**\n"
        for alert in alerts:
            summary += f"> {alert}\n"
        summary += "\n---\n"
    else:
        summary += "> ✅ **No critical alerts** - Mandate portfolio appears well-managed\n\n---\n"
    
    return df_final, summary

def create_debit_mandates_app():
    """Create Gradio app for debit mandates display"""
    with gr.Blocks(title="🏦 Debit Mandates Dashboard") as app:
        gr.Markdown("# 🏦 **DEBIT MANDATES DASHBOARD**")
        gr.Markdown("### 💳 Comprehensive debit order mandate management and analysis")
        
        with gr.Row():
            with gr.Column(scale=2):
                user_id_input = gr.Textbox(
                    label="🆔 Client ID", 
                    placeholder="Enter client ID...",
                    value="10003"  # Using the example client_id from your data
                )
            with gr.Column(scale=1):
                search_button = gr.Button("🚀 Search Mandates", variant="primary", size="lg")
        
        # Summary section
        summary_output = gr.Markdown()
        
        # Mandates table
        gr.Markdown("### 📋 **Mandate Details**")
        dataframe_output = gr.Dataframe(
            headers=[
                "Created Date", "Created Time", "First Collection", "State", "Service Type", 
                "Frequency", "Collection Amount", "Maximum Amount", "Auth Status", 
                "Account Info", "Days to Collection", "Mandate ID"
            ],
            max_height=600,
            show_search='filter',
            wrap=False,
            interactive=False
        )
        
        def update_debit_mandates(user_id):
            df, summary = display_client_debit_mandates(user_id)
            return summary, df
        
        # Event handlers
        search_button.click(
            fn=update_debit_mandates, 
            inputs=user_id_input, 
            outputs=[summary_output, dataframe_output]
        )
        
        user_id_input.submit(
            fn=update_debit_mandates, 
            inputs=user_id_input, 
            outputs=[summary_output, dataframe_output]
        )
        
        # Load initial data
        initial_df, initial_summary = display_client_debit_mandates("10003")
        summary_output.value = initial_summary
        dataframe_output.value = initial_df
    
    return app

# Alternative function for integration into existing dashboard
def get_debit_mandates_summary(user_id):
    """Get a concise debit mandates summary for dashboard integration"""
    data = CartrackSQLDatabase.get_client_debit_mandates.invoke(user_id)
    if not data:
        return "❌ No debit mandates available"
    
    df = pd.DataFrame(data)
    total = len(df)
    active = len(df[df['debicheck_mandate_state'] == 'Active'])
    created = len(df[df['debicheck_mandate_state'] == 'Created'])
    authenticated = len(df[df['authenticated'] == True])
    total_amount = df['collection_amount'].astype(float).sum()
    
    return f"""
### 🏦 **Debit Mandates Summary**
> 📊 **{total} total mandates** | 🟢 **{active} active** | 🟡 **{created} created**
> ✅ **{authenticated} authenticated** | 💰 **R {total_amount:,.2f} total value**
"""

# Usage example - you can integrate this into your main app
if __name__ == "__main__":
    demo = create_debit_mandates_app()
    demo.launch()