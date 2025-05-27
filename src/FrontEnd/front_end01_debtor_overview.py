import gradio as gr
import sys
sys.path.append("../..")
from src.Database import CartrackSQLDatabase

def format_currency(value):
    """Format currency values"""
    try:
        return f"R {float(value):,.2f}"
    except:
        return f"R {value}"

def get_filtered_data(user_id=None, client_type="Individual", min_age=0, max_age=30, 
                     min_balance=100, payment_arrangement="Active", invoice_option="All", 
                     pre_legal=False, rejected=False, batch_size=20):
    """Get filtered data from database"""
    filters = {
        'number_records': batch_size,
        'user_id': user_id if user_id and user_id > 0 else None,
        'client_type': client_type if client_type != "All" else None,
        'min_age_days': min_age,
        'max_age_days': max_age,
        'min_balance_total': min_balance,
        'payment_arrangement_status': payment_arrangement if payment_arrangement != "All" else None,
        'invoice_option': invoice_option if invoice_option != "All" else None,
        'pre_legal': pre_legal,
        'rejected': rejected,
        'sort_by': "balance_total",
        'sort_direction': "DESC",
    }
    return CartrackSQLDatabase.get_debtor_age_analysis.invoke(filters)

def format_table_data(results):
    """Format results for table display"""
    table_data = []
    for client in results:
        # Build status badges
        status_badges = []
        if client.get('payment_arrangement') == "Active":
            status_badges.append("Payment Plan")
        if client.get('rejected', 'No') == 'Yes':
            status_badges.append("Rejected")
        if client.get('pre_legal_billing', 'No') == 'Yes':
            status_badges.append("Pre-Legal")
        
        status_str = " | ".join(status_badges) if status_badges else "Normal"
        
        table_data.append([
            client.get('full_name', 'Unknown'),
            f"{client.get('user_id', 'N/A')} | {client.get('username', 'N/A')}",
            client.get('client_type', 'N/A'),
            format_currency(client.get('balance_total', '0')),
            format_currency(client.get('balance_current', '0')),
            format_currency(client.get('balance_30_days', '0')),
            format_currency(client.get('balance_60_days', '0')),
            format_currency(client.get('balance_90_days', '0')),
            format_currency(client.get('balance_120_days', '0')),
            f"{client.get('age_days', '0')} days",
            str(client.get('vehicle_count', '0')),
            format_currency(client.get('annual_recurring_revenue_excl_vat', '0')),
            format_currency(client.get('crv', '0')),
            client.get('invoice_option', 'N/A'),
            client.get('payment_arrangement', 'None'),
            client.get('branch_name', 'N/A'),
            client.get('source', 'N/A'),
            status_str
        ])
    return table_data

def search_clients(user_id, client_type, min_age, max_age, min_balance, 
                  payment_arrangement, invoice_option, pre_legal, rejected, batch_size=20):
    """Search clients with filters"""
    try:
        results = get_filtered_data(user_id, client_type, min_age, max_age, min_balance,
                                  payment_arrangement, invoice_option, pre_legal, rejected, batch_size)
        table_data = format_table_data(results)
        summary = f"✅ Found {len(results)} clients matching criteria"
        show_load_more = len(results) >= batch_size
        return table_data, summary, results, gr.update(visible=show_load_more)
    except Exception as e:
        return [], f"❌ Search failed: {str(e)}", [], gr.update(visible=False)

def quick_search(user_id, batch_size=20):
    """Quick search by client ID only"""
    if not user_id or user_id <= 0:
        return search_clients(None, "All", 0, 365, 0, "All", "All", False, False, batch_size)
    return search_clients(user_id, "All", 0, 365, 0, "All", "All", False, False, batch_size)

def load_more_results(current_results, user_id, client_type, min_age, max_age, min_balance,
                     payment_arrangement, invoice_option, pre_legal, rejected):
    """Load more results by increasing batch size"""
    try:
        new_batch_size = len(current_results) + 20
        results = get_filtered_data(user_id, client_type, min_age, max_age, min_balance,
                                  payment_arrangement, invoice_option, pre_legal, rejected, new_batch_size)
        table_data = format_table_data(results)
        summary = f"✅ Found {len(results)} clients matching criteria"
        show_load_more = len(results) >= new_batch_size
        
        return table_data, summary, results, gr.update(visible=show_load_more)
    except Exception as e:
        return [], f"❌ Load more failed: {str(e)}", current_results, gr.update(visible=False)

def show_client_details(evt: gr.SelectData, current_results):
    """Show client details when table row is selected"""
    try:
        if not current_results or evt.index[0] >= len(current_results):
            return create_empty_details()
        
        client = current_results[evt.index[0]]
        client_id = client.get('user_id', None)
        
        basic_info = f"""#### 👤 **Basic Information**

**{client.get('full_name', 'Unknown Client')}**
- **🆔 ID:** {client.get('user_id', 'N/A')}
- **👤 Username:** {client.get('username', 'N/A')}
- **📋 Type:** {client.get('client_type', 'N/A')}
- **🚗 Vehicles:** {client.get('vehicle_count', '0')}
- **📅 First Contract:** {client.get('first_contract_start_date', 'N/A')}
"""
        
        financial_info = f"""#### 💰 **Financial Overview**

- **💳 Total Balance:** {format_currency(client.get('balance_total', '0'))}
- **📈 ARR (excl. VAT):** {format_currency(client.get('annual_recurring_revenue_excl_vat', '0'))}
- **💰 CRV:** {format_currency(client.get('crv', '0'))}
- **📋 Unbilled:** {format_currency(client.get('unbilled_subscription', '0'))}
"""
        
        aging_info = f"""#### ⏰ **Aging Analysis**

- **🟢 Current:** {format_currency(client.get('balance_current', '0'))}
- **🟡 30 Days:** {format_currency(client.get('balance_30_days', '0'))}
- **🟠 60 Days:** {format_currency(client.get('balance_60_days', '0'))}
- **🔴 90 Days:** {format_currency(client.get('balance_90_days', '0'))}
- **⚠️ 120+ Days:** {format_currency(client.get('balance_120_days', '0'))}
- **📅 Age:** {client.get('age_days', '0')} days
"""
        
        account_info = f"""#### 📊 **Account Details**

- **💳 Invoice Option:** {client.get('invoice_option', 'N/A')}
- **📋 Payment Arrangement:** {client.get('payment_arrangement', 'None')}
- **🏢 Branch:** {client.get('branch_name', 'N/A')}
- **📍 Source:** {client.get('source', 'N/A')}
- **⚖️ Pre-Legal:** {'✅ Yes' if client.get('pre_legal_billing') == 'Yes' else '❌ No'}
- **❌ Rejected:** {'🔴 Yes' if client.get('rejected') == 'Yes' else '✅ No'}
"""
        
        return basic_info, financial_info, aging_info, account_info, gr.update(visible=True), client_id
    except:
        return create_empty_details() + (None,)

def create_empty_details():
    """Create empty client details"""
    return (
        "## 👤 **Basic Information**\n\nClick on a table row to view details...",
        "## 💰 **Financial Overview**\n\n", 
        "## ⏰ **Aging Analysis**\n\n", 
        "## 📊 **Account Details**\n\n",
        gr.update(visible=False)
    )

def reset_filters():
    """Reset all filters to defaults and load initial data"""
    table_data, summary, results, load_more_btn = search_clients(None, "Individual", 0, 30, 100, "Active", "All", False, False)
    basic, financial, aging, account, view_btn = create_empty_details()
    return (table_data, summary, results, load_more_btn, basic, financial, aging, account, view_btn,
            0, "Individual", 0, 30, 100, "Active", "All", False, False)

def navigate_to_client_details(client_id):
    """Navigate to client details tab and search for client"""
    if client_id:
        return gr.update(selected=1), str(client_id)
    return gr.update(), ""

def create_client_overview_interface():
    """Create client overview interface"""
    
    # State for current results
    current_results = gr.State([])
    selected_client_id = gr.State(None)

    # Load initial data before creating the interface
    try:
        initial_results = get_filtered_data(None, "Individual", 0, 30, 100, "Active", "All", False, False, 20)
        initial_table_data = format_table_data(initial_results)
        initial_summary = f"✅ Found {len(initial_results)} clients (initial load)"
        initial_show_load_more = len(initial_results) >= 20
    except Exception as e:
        initial_table_data = []
        initial_results = []
        initial_summary = f"❌ Failed to load initial data: {str(e)}"
        initial_show_load_more = False

    with gr.Row():
        # Left Sidebar - Search Panel
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### 🔍 Search & Filter")
            
            # Quick Search
            gr.Markdown("#### Quick Search")
            user_id_input = gr.Number(
                label="Client ID",
                minimum=0,
                maximum=999999,
                step=1
            )
            quick_search_btn = gr.Button("🔎 Quick Search", variant="primary", size="sm")
            
            # Advanced Filters
            with gr.Accordion("🔧 Advanced Filters", open=False):
                client_type = gr.Dropdown(
                    choices=["All", "Individual", "Company", "Finance Agent", "Fitment Center", "Business Source"],
                    value="Individual",
                    label="Client Type"
                )
                
                with gr.Row():
                    min_age_days = gr.Number(label="Min Age", value=0, minimum=0, step=1)
                    max_age_days = gr.Number(label="Max Age", value=30, minimum=0, step=1)
                
                min_balance_total = gr.Slider(
                    minimum=0, maximum=50000, value=100, step=100,
                    label="Minimum Balance (R)"
                )
                
                payment_arrangement_status = gr.Dropdown(
                    choices=["All", "Active", "Finished", "No"],
                    value="Active",
                    label="Payment Arrangement"
                )
                
                invoice_option = gr.Dropdown(
                    choices=["All", "Direct Debit", "Credit Card", "Invoice"],
                    value="All",
                    label="Invoice Option"
                )
                
                with gr.Row():
                    pre_legal = gr.Checkbox(label="Pre-Legal Only", value=False)
                    rejected = gr.Checkbox(label="Rejected Only", value=False)
            
            with gr.Row():
                search_btn = gr.Button("🔍 Apply Filters", variant="primary", size="sm")
                reset_btn = gr.Button("🔄 Reset", variant="secondary", size="sm")
            
            load_more_btn = gr.Button("📄 Load More Results", variant="secondary", visible=initial_show_load_more, size="sm")
            results_info = gr.Markdown(initial_summary)
        
        # Main Content - Results Table
        with gr.Column(scale=5):
            gr.Markdown("### 📋 Search Results")
            
            results_table = gr.Dataframe(
                headers=[
                    "Client Name", "ID | Username", "Type", "Balance Total", "Current", 
                    "30 Days", "60 Days", "90 Days", "120+ Days", "Age (Days)", 
                    "Vehicles Count", "ARR (excl VAT)", "CRV", "Invoice Option", 
                    "Payment Arrangement", "Branch", "Source", "Status"
                ],
                datatype=["str"] * 18,
                column_widths=['230px', '170px', '100px', '100px', '100px', 
                            '100px', '100px', '100px', '100px', '100px', 
                                '120px', '135px', '135px', '176px', 
                            '161px', '177px', '550px', '200px'],
                wrap=False,
                interactive=False,
                value=initial_table_data,  # Load initial data
                show_label=False,
                max_height=1400,
                row_count=(1, "dynamic"),
                show_row_numbers=True
            )
        
        # Right Sidebar - Client Details
        with gr.Column(scale=2, min_width=200):
            gr.Markdown("### 👤 Client Summary")
            
            view_profile_btn = gr.Button(
                "👤 View Full Details", variant="primary", size="sm", visible=False
            )
            
            client_basic_info = gr.Markdown("#### 👤 **Basic Information**\n\nClick on a table row to view details...")
            client_financial_info = gr.Markdown("#### 💰 **Financial Overview**\n\n")
            client_aging_info = gr.Markdown("#### ⏰ **Aging Analysis**\n\n")
            client_account_info = gr.Markdown("#### 📊 **Account Details**\n\n")
    
    # Define input/output groups
    search_inputs = [user_id_input, client_type, min_age_days, max_age_days, min_balance_total,
                    payment_arrangement_status, invoice_option, pre_legal, rejected]
    
    search_outputs = [results_table, results_info, current_results, load_more_btn]
    detail_outputs = [client_basic_info, client_financial_info, client_aging_info, 
                     client_account_info, view_profile_btn, selected_client_id]
    
    # Set initial state
    current_results.value = initial_results
    
    # Event handlers
    search_btn.click(fn=search_clients, inputs=search_inputs, outputs=search_outputs)
    quick_search_btn.click(fn=quick_search, inputs=[user_id_input], outputs=search_outputs)
    user_id_input.submit(fn=quick_search, inputs=[user_id_input], outputs=search_outputs)
    
    load_more_btn.click(
        fn=load_more_results,
        inputs=[current_results] + search_inputs,
        outputs=search_outputs
    )
    
    results_table.select(fn=show_client_details, inputs=[current_results], outputs=detail_outputs)
    reset_btn.click(fn=reset_filters, outputs=search_outputs + detail_outputs + search_inputs)
    
    return view_profile_btn, selected_client_id



