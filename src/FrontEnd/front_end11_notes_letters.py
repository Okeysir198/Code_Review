import gradio as gr
import pandas as pd
import sys
import datetime
sys.path.append("../..")
from src.Database import CartrackSQLDatabase

def display_client_notes_and_letters(user_id=None):
    """Display comprehensive client notes and debtor letters using DataFrames + summary"""
    if not user_id:
        return None, None, "Please enter a client ID and click Search to view notes and letters."
    
    # Helper function to calculate days ago with timezone handling
    def calculate_days_ago(date_str):
        try:
            date_obj = pd.to_datetime(date_str)
            if date_obj.tz is not None:
                date_obj = date_obj.tz_localize(None)
            return (datetime.datetime.now() - date_obj).days
        except:
            return "Unknown"
    
    # Get client notes data
    notes_data = CartrackSQLDatabase.get_client_notes.invoke(user_id)
    
    # Get debtor letters data
    letters_data = CartrackSQLDatabase.get_client_debtor_letters.invoke(user_id)
    
    # Process Notes DataFrame
    df_notes = pd.DataFrame()
    if notes_data:
        df_notes_display = pd.DataFrame(notes_data)
        
        # Format dates for notes
        df_notes_display['Date'] = pd.to_datetime(df_notes_display['timestamp']).dt.strftime('%Y-%m-%d')
        df_notes_display['Time'] = pd.to_datetime(df_notes_display['timestamp']).dt.strftime('%H:%M')
        df_notes_display['Days Ago'] = df_notes_display['timestamp'].apply(calculate_days_ago)
        
        # Format source with emojis
        source_map = {
            'Client': '👤 Client',
            'SMS': '📱 SMS',
            'System': '⚙️ System',
            'Email': '📧 Email',
            'Phone': '📞 Phone',
            'Portal': '🌐 Portal'
        }
        df_notes_display['Source'] = df_notes_display['source'].map(source_map).fillna('📋 ' + df_notes_display['source'].astype(str))
        
        # Format created by
        creator_map = {
            'SYSTEM': '🤖 System',
            'divanv': '👤 divanv',
            'admin': '👤 Admin',
            'ai_agent': '🤖 AI Agent'
        }
        df_notes_display['Created By'] = df_notes_display['created_by'].map(creator_map).fillna('👤 ' + df_notes_display['created_by'].astype(str))
        
        # Truncate long notes for display
        df_notes_display['Note Content'] = df_notes_display['note'].apply(
            lambda x: str(x)[:80] + '...' if len(str(x)) > 80 else str(x)
        )
        
        # Categorize notes by content type
        def categorize_note(note_text):
            note_lower = str(note_text).lower()
            if 'sms sent' in note_lower:
                return '📱 SMS Notification'
            elif 'email' in note_lower:
                return '📧 Email Communication'
            elif 'fitment' in note_lower:
                return '🔧 Fitment Related'
            elif 'test' in note_lower:
                return '🧪 System Test'
            elif 'vat' in note_lower:
                return '💰 Financial Notice'
            elif 'portal' in note_lower:
                return '🌐 Portal Related'
            else:
                return '📝 General Note'
        
        df_notes_display['Category'] = df_notes_display['note'].apply(categorize_note)
        
        # Select columns for notes display
        df_notes = df_notes_display[['Date', 'Time', 'Days Ago', 'Category', 'Source', 'Created By', 'Note Content']].sort_values('Date', ascending=False)
    
    # Process Letters DataFrame
    df_letters = pd.DataFrame()
    if letters_data:
        df_letters_display = pd.DataFrame(letters_data)
        
        # Format dates for letters
        df_letters_display['Sent Date'] = pd.to_datetime(df_letters_display['sent_date']).dt.strftime('%Y-%m-%d')
        df_letters_display['Sent Time'] = pd.to_datetime(df_letters_display['sent_date']).dt.strftime('%H:%M')
        df_letters_display['Days Ago'] = df_letters_display['sent_date'].apply(calculate_days_ago)
        
        # Categorize letter types
        def categorize_letter(report_name):
            name_lower = str(report_name).lower()
            if 'overdue' in name_lower:
                return '⚠️ Overdue Notice'
            elif 'reminder' in name_lower:
                return '🔔 Payment Reminder'
            elif 'final' in name_lower:
                return '🚨 Final Notice'
            elif 'statement' in name_lower:
                return '📊 Statement'
            elif 'demand' in name_lower:
                return '📋 Demand Letter'
            else:
                return '📄 General Letter'
        
        df_letters_display['Letter Type'] = df_letters_display['report_name'].apply(categorize_letter)
        
        # Format report name for display
        df_letters_display['Report Name'] = df_letters_display['report_name'].apply(
            lambda x: str(x).replace('.rpt', '').replace('_', ' ').title()
        )
        
        # Add urgency indicator
        def get_urgency(report_name, days_ago):
            if isinstance(days_ago, int):
                if 'final' in str(report_name).lower() or 'demand' in str(report_name).lower():
                    return '🚨 High Priority'
                elif 'overdue' in str(report_name).lower():
                    return '⚠️ Medium Priority'
                elif days_ago <= 7:
                    return '🔔 Recent'
                else:
                    return '📝 Standard'
            return '📝 Standard'
        
        df_letters_display['Priority'] = df_letters_display.apply(
            lambda row: get_urgency(row['report_name'], row['Days Ago']), axis=1
        )
        
        # Select columns for letters display
        df_letters = df_letters_display[['Sent Date', 'Sent Time', 'Days Ago', 'Letter Type', 'Priority', 'Report Name', 'report_id']].rename(columns={
            'report_id': 'Report ID'
        }).sort_values('Sent Date', ascending=False)
    
    # Calculate summary statistics
    total_notes = len(notes_data) if notes_data else 0
    total_letters = len(letters_data) if letters_data else 0
    
    # Notes analysis
    if notes_data:
        notes_df_temp = pd.DataFrame(notes_data)
        sms_count = len(notes_df_temp[notes_df_temp['source'] == 'SMS'])
        client_count = len(notes_df_temp[notes_df_temp['source'] == 'Client'])
        system_count = len(notes_df_temp[notes_df_temp['created_by'] == 'SYSTEM'])
        
        # Recent activity (last 30 days)
        recent_notes = []
        for note in notes_data:
            days_ago = calculate_days_ago(note['timestamp'])
            if isinstance(days_ago, int) and days_ago <= 30:
                recent_notes.append(note)
        recent_notes_count = len(recent_notes)
        
        # Date range for notes
        earliest_note = pd.to_datetime(notes_df_temp['timestamp']).min().strftime('%Y-%m-%d')
        latest_note = pd.to_datetime(notes_df_temp['timestamp']).max().strftime('%Y-%m-%d')
    else:
        sms_count = client_count = system_count = recent_notes_count = 0
        earliest_note = latest_note = "N/A"
    
    # Letters analysis
    if letters_data:
        letters_df_temp = pd.DataFrame(letters_data)
        
        # Recent letters (last 30 days)
        recent_letters = []
        for letter in letters_data:
            days_ago = calculate_days_ago(letter['sent_date'])
            if isinstance(days_ago, int) and days_ago <= 30:
                recent_letters.append(letter)
        recent_letters_count = len(recent_letters)
        
        # Date range for letters
        earliest_letter = pd.to_datetime(letters_df_temp['sent_date']).min().strftime('%Y-%m-%d')
        latest_letter = pd.to_datetime(letters_df_temp['sent_date']).max().strftime('%Y-%m-%d')
        
        # Letter type breakdown
        letter_types = {}
        for letter in letters_data:
            letter_type = categorize_letter(letter['report_name'])
            letter_types[letter_type] = letter_types.get(letter_type, 0) + 1
    else:
        recent_letters_count = 0
        earliest_letter = latest_letter = "N/A"
        letter_types = {}
    
    # Build comprehensive summary
    summary = f"""
# 📋 **Client Communication History - Client {user_id}**

## 📊 **Communication Overview**
> 📝 **Total Notes:** `{total_notes}` | 📄 **Total Letters:** `{total_letters}` | 🕒 **Recent Activity (30 days):** `{recent_notes_count + recent_letters_count}`

| 📱 **SMS Notes** | 👤 **Client Notes** | 🤖 **System Notes** | 📄 **Recent Letters** |
|:---:|:---:|:---:|:---:|
| `{sms_count}` | `{client_count}` | `{system_count}` | `{recent_letters_count}` |

## 📅 **Timeline Coverage**
> 📝 **Notes Period:** `{earliest_note}` to `{latest_note}`
> 📄 **Letters Period:** `{earliest_letter}` to `{latest_letter}`

## 📄 **Letter Type Breakdown**
"""
    
    # Add letter types breakdown
    if letter_types:
        for letter_type, count in letter_types.items():
            summary += f"> {letter_type}: `{count} letters`\n"
    else:
        summary += "> 📭 **No debtor letters found**\n"
    
    summary += f"""

## 📅 **Report Information**
> 📅 **Generated:** `{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}` | 🔄 **Data Sources:** Client Notes + Debtor Letters

---
"""
    
    # Add intelligent alerts
    alerts = []
    
    # Recent activity alerts
    if recent_letters_count > 0:
        alerts.append(f"📄 **Recent debtor letters** - {recent_letters_count} letters sent in last 30 days")
    
    if recent_notes_count > 5:
        alerts.append(f"📝 **High note activity** - {recent_notes_count} notes added in last 30 days")
    
    # Communication pattern alerts
    if total_letters > 0:
        if total_letters >= 3:
            alerts.append(f"⚠️ **Multiple debtor letters** - {total_letters} letters indicate collection activity")
        
        # Check for escalation pattern
        overdue_letters = [l for l in letters_data if 'overdue' in l['report_name'].lower()]
        if len(overdue_letters) > 0:
            alerts.append(f"🚨 **Overdue notices sent** - {len(overdue_letters)} overdue letters indicate payment issues")
    
    # SMS communication patterns
    if sms_count > 10:
        alerts.append(f"📱 **High SMS volume** - {sms_count} SMS communications recorded")
    
    # System vs manual note ratio
    if total_notes > 0:
        manual_notes = total_notes - system_count
        if manual_notes == 0 and total_notes > 5:
            alerts.append("🤖 **All automated notes** - No manual client interaction notes found")
        elif manual_notes > system_count:
            alerts.append(f"👤 **High manual interaction** - {manual_notes} manual notes vs {system_count} system notes")
    
    # Check for fitment or technical issues
    if notes_data:
        fitment_notes = [n for n in notes_data if 'fitment' in n['note'].lower()]
        if fitment_notes:
            alerts.append(f"🔧 **Technical issues noted** - {len(fitment_notes)} fitment-related notes")
    
    # Long period without communication
    if notes_data:
        last_note_days = calculate_days_ago(notes_data[0]['timestamp'])  # Most recent note
        if isinstance(last_note_days, int) and last_note_days > 365:
            alerts.append(f"📅 **Long silence period** - Last note was {last_note_days} days ago")
    
    if alerts:
        summary += "### 🚨 **Communication Alerts & Insights**\n"
        for alert in alerts:
            summary += f"> {alert}\n"
        summary += "\n---\n"
    else:
        summary += "> ✅ **No critical communication alerts** - Communication history appears normal\n\n---\n"
    
    return df_notes, df_letters, summary

def create_client_notes_letters_app():
    """Create Gradio app for client notes and debtor letters display"""
    with gr.Blocks(title="📋 Client Communication Dashboard") as app:
        gr.Markdown("# 📋 **CLIENT COMMUNICATION DASHBOARD**")
        gr.Markdown("### 📝 Comprehensive notes and debtor letters analysis")
        
        with gr.Row():
            with gr.Column(scale=2):
                user_id_input = gr.Textbox(
                    label="🆔 Client ID", 
                    placeholder="Enter client ID...",
                    value="28173"  # Default example
                )
            with gr.Column(scale=1):
                search_button = gr.Button("🚀 Search Communications", variant="primary", size="lg")
        
        # Summary section
        summary_output = gr.Markdown()
        
        # Notes table
        gr.Markdown("### 📝 **Client Notes History**")
        notes_dataframe = gr.Dataframe(
            headers=["Date", "Time", "Days Ago", "Category", "Source", "Created By", "Note Content"],
            max_height=400,
            show_search='filter',
            wrap=False,
            interactive=False
        )
        
        # Letters table  
        gr.Markdown("### 📄 **Debtor Letters History**")
        letters_dataframe = gr.Dataframe(
            headers=["Sent Date", "Sent Time", "Days Ago", "Letter Type", "Priority", "Report Name", "Report ID"],
            max_height=300,
            show_search='filter',
            wrap=False,
            interactive=False
        )
        
        def update_communications(user_id):
            df_notes, df_letters, summary = display_client_notes_and_letters(user_id)
            return summary, df_notes, df_letters
        
        # Event handlers
        search_button.click(
            fn=update_communications, 
            inputs=user_id_input, 
            outputs=[summary_output, notes_dataframe, letters_dataframe]
        )
        
        user_id_input.submit(
            fn=update_communications, 
            inputs=user_id_input, 
            outputs=[summary_output, notes_dataframe, letters_dataframe]
        )
        
        # Load initial data
        initial_notes, initial_letters, initial_summary = display_client_notes_and_letters("28173")
        summary_output.value = initial_summary
        notes_dataframe.value = initial_notes
        letters_dataframe.value = initial_letters
    
    return app

# Alternative functions for integration into existing dashboard
def get_client_notes_summary(user_id):
    """Get a concise client notes summary for dashboard integration"""
    notes_data = CartrackSQLDatabase.get_client_notes.invoke(user_id)
    if not notes_data:
        return "❌ No client notes available"
    
    total = len(notes_data)
    sms_count = len([n for n in notes_data if n['source'] == 'SMS'])
    client_count = len([n for n in notes_data if n['source'] == 'Client'])
    
    # Recent activity (last 30 days)
    def calculate_days_ago(date_str):
        try:
            date_obj = pd.to_datetime(date_str)
            if date_obj.tz is not None:
                date_obj = date_obj.tz_localize(None)
            return (datetime.datetime.now() - date_obj).days
        except:
            return 999
    
    recent_count = len([n for n in notes_data if calculate_days_ago(n['timestamp']) <= 30])
    
    return f"""
### 📝 **Client Notes Summary**
> 📊 **{total} total notes** | 📱 **{sms_count} SMS** | 👤 **{client_count} client notes**
> 🕒 **{recent_count} recent** (last 30 days)
"""

def get_debtor_letters_summary(user_id):
    """Get a concise debtor letters summary for dashboard integration"""
    letters_data = CartrackSQLDatabase.get_client_debtor_letters.invoke(user_id)
    if not letters_data:
        return "✅ No debtor letters sent"
    
    total = len(letters_data)
    overdue_count = len([l for l in letters_data if 'overdue' in l['report_name'].lower()])
    
    # Most recent letter
    most_recent = max(letters_data, key=lambda x: x['sent_date'])
    try:
        days_ago = (datetime.datetime.now() - pd.to_datetime(most_recent['sent_date']).tz_localize(None)).days
    except:
        days_ago = "Unknown"
    
    return f"""
### 📄 **Debtor Letters Summary**
> 🚨 **{total} letters sent** | ⚠️ **{overdue_count} overdue notices**
> 🕒 **Last letter:** {days_ago} days ago
"""

# Usage example - you can integrate this into your main app
if __name__ == "__main__":
    demo = create_client_notes_letters_app()
    demo.launch()