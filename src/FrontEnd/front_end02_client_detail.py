import gradio as gr
import sys
import asyncio
import time
from functools import lru_cache
from typing import Tuple, Any, Optional
from dataclasses import dataclass

sys.path.append("../..")

from src.Database import CartrackSQLDatabase
from app_config import CONFIG
from src.VoiceHandler import VoiceInteractionHandler
from src.Agents import react_agent_graph
from src.Agents.call_center_agent.data.client_data_fetcher import get_client_data

from .front_end03_voice_chat import create_voice_chat_block
from .front_end04_profile_display import display_client_profile
from .front_end05_account_display import display_client_account_overview
from .front_end06_contracts_display import display_client_contracts
from .front_end07_client_account_statement import display_account_statement
from .front_end08_banking_details import display_client_banking_details
from .front_end09_payment_history import display_payment_history
from .front_end10_debit_mandates import display_client_debit_mandates
from .front_end11_notes_letters import display_client_notes_and_letters

# Configuration
CACHE_SIZE = 20
CACHE_DURATION = 300  # 5 minutes
ASYNC_TIMEOUT = 50.0

@dataclass
class FunctionConfig:
    """Configuration for display functions"""
    func: callable
    expected_returns: int
    name: str


class ClientDataCache:
    """Centralized caching system for client data"""
    
    def __init__(self, cache_size: int = CACHE_SIZE):
        self.cache_size = cache_size
        self._setup_cached_functions()
    
    def _setup_cached_functions(self):
        """Setup all cached functions"""
        self.cached_profile = lru_cache(maxsize=self.cache_size)(self._cache_wrapper(display_client_profile))
        self.cached_account = lru_cache(maxsize=self.cache_size)(self._cache_wrapper(display_client_account_overview))
        self.cached_contracts = lru_cache(maxsize=self.cache_size)(self._cache_wrapper(display_client_contracts))
        self.cached_statement = lru_cache(maxsize=self.cache_size)(self._cache_wrapper(display_account_statement))
        self.cached_banking = lru_cache(maxsize=self.cache_size)(self._cache_wrapper(display_client_banking_details))
        self.cached_payments = lru_cache(maxsize=self.cache_size)(self._cache_wrapper(display_payment_history))
        self.cached_mandates = lru_cache(maxsize=self.cache_size)(self._cache_wrapper(display_client_debit_mandates))
        self.cached_notes = lru_cache(maxsize=self.cache_size)(self._cache_wrapper(display_client_notes_and_letters))
    
    @staticmethod
    def _cache_wrapper(func):
        """Wrapper to add cache key parameter to functions"""
        def wrapper(user_id: str, cache_key: int = None):
            return func(user_id)
        return wrapper
    
    @staticmethod
    def get_cache_key() -> int:
        """Generate cache key - refreshes every 5 minutes"""
        return int(time.time() // CACHE_DURATION)


class DataLoader:
    """Handles safe data loading with error handling"""
    
    def __init__(self, cache: ClientDataCache):
        self.cache = cache
        self.function_configs = [
            FunctionConfig(cache.cached_profile, 2, "profile"),
            FunctionConfig(cache.cached_account, 7, "account"),
            FunctionConfig(cache.cached_contracts, 2, "contracts"),
            FunctionConfig(cache.cached_statement, 2, "statement"),
            FunctionConfig(cache.cached_banking, 1, "banking"),
            FunctionConfig(cache.cached_payments, 3, "payments"),
            FunctionConfig(cache.cached_mandates, 2, "mandates"),
            FunctionConfig(cache.cached_notes, 3, "notes"),
        ]
    
    def safe_function_call(self, func: callable, user_id: str, expected_returns: int) -> Tuple:
        """Safely call a function and ensure it returns the expected number of values"""
        try:
            cache_key = self.cache.get_cache_key()
            result = func(user_id, cache_key)
            
            if isinstance(result, tuple) and len(result) == expected_returns:
                return result
            elif not isinstance(result, tuple) and expected_returns == 1:
                return (result,)
            else:
                return self._create_error_tuple(expected_returns, f"Function returned {type(result)} instead of {expected_returns} values")
        except Exception as e:
            return self._create_error_tuple(expected_returns, str(e))
    
    @staticmethod
    def _create_error_tuple(expected_returns: int, error_msg: str) -> Tuple:
        """Create error tuple with appropriate number of None/error values"""
        return tuple([None if i < expected_returns - 1 else f"Error: {error_msg}" for i in range(expected_returns)])
    
    async def load_all_data_async(self, user_id: str) -> Tuple:
        """Load all tabs in parallel with caching"""
        if not user_id:
            return self._create_empty_response()
        
        try:
            tasks = [
                asyncio.to_thread(self.safe_function_call, config.func, user_id, config.expected_returns)
                for config in self.function_configs
            ]
            
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), 
                timeout=ASYNC_TIMEOUT
            )
            
            return self._process_results(results)
            
        except Exception as e:
            return self._create_error_response(str(e))
    
    def _create_empty_response(self) -> Tuple:
        """Create empty response when no user_id provided"""
        empty_msg = "Please enter a client ID to search."
        empty_df = None
        return (
            empty_df, empty_msg,  # Profile
            empty_df, empty_df, empty_df, empty_df, empty_df, empty_msg, empty_msg,  # Account
            empty_df, empty_msg,  # Contracts
            empty_df, empty_msg,  # Statement
            empty_msg,  # Banking
            empty_df, empty_df, empty_msg,  # Payment History
            empty_df, empty_msg,  # Debit Mandates
            empty_df, empty_df, empty_msg  # Notes & Letters
        )
    
    def _create_error_response(self, error_msg: str) -> Tuple:
        """Create error response"""
        return (
            None, f"Error: {error_msg}",  # Profile
            None, None, None, None, None, f"Error: {error_msg}", f"Error: {error_msg}",  # Account
            None, f"Error: {error_msg}",  # Contracts
            None, f"Error: {error_msg}",  # Statement
            f"Error: {error_msg}",  # Banking
            None, None, f"Error: {error_msg}",  # Payment History
            None, f"Error: {error_msg}",  # Debit Mandates
            None, None, f"Error: {error_msg}"  # Notes & Letters
        )
    
    def _process_results(self, results) -> Tuple:
        """Process async results and handle exceptions"""
        final_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                expected_returns = self.function_configs[i].expected_returns
                error_values = self._create_error_tuple(expected_returns, str(result))
                final_results.extend(error_values)
            else:
                final_results.extend(result)
        
        return tuple(final_results)


class UIBuilder:
    """Handles UI component creation and initialization"""
    
    def __init__(self, data_loader: DataLoader, initial_client_id: str = "28173"):
        self.data_loader = data_loader
        self.initial_client_id = initial_client_id
    
    def create_search_panel(self) -> Tuple[gr.Textbox, gr.Button, gr.Markdown]:
        """Create the search input panel"""
        with gr.Row():
            with gr.Column(scale=2):
                user_id_input = gr.Textbox(
                    label="🆔 Client ID",
                    placeholder="Enter client ID to search...",
                    value=self.initial_client_id,
                    container=True
                )
            with gr.Column(scale=0, min_width=200):
                search_button = gr.Button(
                    "🚀 Search Client",
                    variant="primary",
                    size="lg"
                )
            with gr.Column(scale=3):
                perf_indicator = gr.Markdown("")
        
        return user_id_input, search_button, perf_indicator
    
    def create_profile_tab(self) -> Tuple[gr.Dataframe, gr.Markdown]:
        """Create profile tab with initial data"""
        vehicles_df, profile_markdown = self.data_loader.safe_function_call(
            self.data_loader.cache.cached_profile, self.initial_client_id, 2
        )
        
        profile_md = gr.Markdown(value=profile_markdown, container=True)
        gr.Markdown("### 🚗 **VEHICLE FLEET**")
        vehicles_table = gr.Dataframe(
            value=vehicles_df,
            headers=["Status", "Vehicle", "Registration", "Color", "Year", "Chassis", "Terminal", "Last Signal"],
            wrap=False,
            interactive=False,
            show_search='filter'
        )
        
        return vehicles_table, profile_md
    
    def create_account_tab(self) -> Tuple:
        """Create account overview tab with initial data"""
        result = self.data_loader.safe_function_call(
            self.data_loader.cache.cached_account, self.initial_client_id, 7
        )
        client_df, status_df, financial_df, billing_df, billing_analysis_df, summary, account_status_md = result
        
        summary_output = gr.Markdown(value=summary)
        status_md_output = gr.Markdown(value=account_status_md)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 👤 **Client Information**")
                client_output = gr.Dataframe(
                    value=client_df,
                    headers=["Field", "Value"],
                    wrap=False,
                    column_widths=["40%", "60%"],
                    interactive=False
                )
                
                gr.Markdown("### 📊 **Account Status**")
                status_output = gr.Dataframe(
                    value=status_df,
                    headers=["Field", "Value"],
                    wrap=False,
                    column_widths=["40%", "60%"],
                    interactive=False
                )
                
                gr.Markdown("### 💰 **Financial Information**")
                financial_output = gr.Dataframe(
                    value=financial_df,
                    headers=["Field", "Value"],
                    wrap=False,
                    column_widths=["40%", "60%"],
                    interactive=False
                )
                
                gr.Markdown("### 💳 **Billing Details**")
                billing_output = gr.Dataframe(
                    value=billing_df,
                    headers=["Field", "Value"],
                    wrap=False,
                    column_widths=["40%", "60%"],
                    interactive=False
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📈 **Billing Analysis**")
                billing_analysis_output = gr.Dataframe(
                    value=billing_analysis_df,
                    headers=["Field", "Value"],
                    wrap=False,
                    column_widths=["40%", "60%"],
                    interactive=False,
                    max_height=2000
                )
        
        return (client_output, status_output, financial_output, billing_output, 
                billing_analysis_output, summary_output, status_md_output)
    
    def create_contracts_tab(self) -> Tuple[gr.Dataframe, gr.Markdown]:
        """Create contracts tab with initial data"""
        contracts_df, summary = self.data_loader.safe_function_call(
            self.data_loader.cache.cached_contracts, self.initial_client_id, 2
        )
        
        summary_output = gr.Markdown(value=summary)
        contracts_output = gr.Dataframe(
            value=contracts_df,
            headers=["Contract ID", "Status", "Sale Date", "Start Date", "End Date", 
                    "Payment Option", "Vehicle ID", "Terminal ID", "Fitter ID", "Package ID", "Branch ID"],
            wrap=False,
            interactive=False,
            show_search='filter'
        )
        
        return contracts_output, summary_output
    
    def create_statement_tab(self) -> Tuple[gr.Dataframe, gr.Markdown]:
        """Create account statement tab with initial data"""
        statement_df, summary = self.data_loader.safe_function_call(
            self.data_loader.cache.cached_statement, self.initial_client_id, 2
        )
        
        summary_output = gr.Markdown(value=summary, container=True)
        statement_output = gr.Dataframe(
            value=statement_df,
            headers=["Date", "Time", "Type", "Status", "Description", "Amount", "Outstanding", "Reference"],
            wrap=False,
            interactive=False,
            show_search='filter'
        )
        
        return statement_output, summary_output
    
    def create_banking_tab(self) -> gr.Markdown:
        """Create banking details tab with initial data"""
        banking_markdown, = self.data_loader.safe_function_call(
            self.data_loader.cache.cached_banking, self.initial_client_id, 1
        )
        
        return gr.Markdown(value=banking_markdown, container=True)
    
    def create_payment_tab(self) -> Tuple[gr.Dataframe, gr.Dataframe, gr.Markdown]:
        """Create payment history tab with initial data"""
        arrangements_df, failed_df, summary = self.data_loader.safe_function_call(
            self.data_loader.cache.cached_payments, "963262", 3  # Using example user_id
        )
        
        summary_output = gr.Markdown(value=summary)
        
        gr.Markdown("### 📊 **Payment Arrangements History**")
        arrangements_output = gr.Dataframe(
            value=arrangements_df,
            headers=["Created Date", "Created Time", "Payment Date", "Status", "State Type",
                    "Payment Type", "Amount", "Created By", "Description", "Additional Info", "Arrangement ID"],
            max_height=400,
            wrap=False,
            interactive=False,
            show_search='filter'
        )
        
        gr.Markdown("### 🚫 **Individual Failed Payments**")
        failed_output = gr.Dataframe(
            value=failed_df,
            headers=["Payment Date", "Failure Reason", "Days Ago"],
            max_height=300,
            show_search='filter',
            wrap=False,
            interactive=False
        )
        
        return arrangements_output, failed_output, summary_output
    
    def create_mandates_tab(self) -> Tuple[gr.Dataframe, gr.Markdown]:
        """Create debit mandates tab with initial data"""
        mandates_df, summary = self.data_loader.safe_function_call(
            self.data_loader.cache.cached_mandates, "10003", 2  # Using example client_id
        )
        
        summary_output = gr.Markdown(value=summary)
        
        gr.Markdown("### 📋 **Mandate Details**")
        mandates_output = gr.Dataframe(
            value=mandates_df,
            headers=["Created Date", "Created Time", "First Collection", "State", "Service Type",
                    "Frequency", "Collection Amount", "Maximum Amount", "Auth Status",
                    "Account Info", "Days to Collection", "Mandate ID"],
            max_height=600,
            show_search='filter',
            wrap=False,
            interactive=False
        )
        
        return mandates_output, summary_output
    
    def create_notes_tab(self) -> Tuple[gr.Dataframe, gr.Dataframe, gr.Markdown]:
        """Create notes and letters tab with initial data"""
        notes_df, letters_df, summary = self.data_loader.safe_function_call(
            self.data_loader.cache.cached_notes, self.initial_client_id, 3
        )
        
        summary_output = gr.Markdown(value=summary)
        
        gr.Markdown("### 📝 **Client Notes History**")
        notes_output = gr.Dataframe(
            value=notes_df,
            headers=["Date", "Time", "Days Ago", "Category", "Source", "Created By", "Note Content"],
            max_height=400,
            show_search='filter',
            wrap=False,
            interactive=False
        )
        
        gr.Markdown("### 📄 **Debtor Letters History**")
        letters_output = gr.Dataframe(
            value=letters_df,
            headers=["Sent Date", "Sent Time", "Days Ago", "Letter Type", "Priority", "Report Name", "Report ID"],
            max_height=300,
            show_search='filter',
            wrap=False,
            interactive=False
        )
        
        return notes_output, letters_output, summary_output


class ClientDashboard:
    """Main dashboard class orchestrating all components"""
    
    def __init__(self, initial_client_id: str = "28173"):
        self.cache = ClientDataCache()
        self.data_loader = DataLoader(self.cache)
        self.ui_builder = UIBuilder(self.data_loader, initial_client_id)
        self.initial_client_id = initial_client_id
    
    def update_all_data(self, user_id: str) -> Tuple:
        """Sync wrapper for async data loading"""
        return asyncio.run(self.data_loader.load_all_data_async(user_id))
    
    def timed_update(self, user_id: str) -> Tuple:
        """Update with performance timing"""
        start_time = time.time()
        try:
            results = self.update_all_data(user_id)
            elapsed = time.time() - start_time
            perf_msg = f"⚡ {elapsed:.2f}s {'🟢' if elapsed < 2 else '🟡' if elapsed < 5 else '🔴'}"
            return results + (perf_msg,)
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            error_response = self.data_loader._create_error_response(str(e))
            return error_response + (error_msg,)
    
    def load_client_data_for_voice(self, user_id: str) -> Optional[dict]:
        """Load client data specifically for voice chat workflow creation."""
        try:
            if not user_id or not user_id.strip():
                return None
            
            # Use the existing data builder to get comprehensive client data
            client_data = get_client_data(user_id)
            return client_data
            
        except Exception as e:
            print(f"Error loading client data for voice chat: {e}")
            return None
    
    def update_voice_client_data(self, user_id: str, voice_chat_block) -> str:
        """Update voice chat with new client data."""
        try:
            if not user_id or not user_id.strip():
                return "❌ No client ID provided"
            
            # Load client data
            client_data = self.load_client_data_for_voice(user_id)
            
            if client_data and hasattr(voice_chat_block, 'interface'):
                # Update voice chat interface with new client data
                status = voice_chat_block.interface.update_client_data(user_id)
                return status
            else:
                return f"❌ Failed to load data for client ID: {user_id}"
                
        except Exception as e:
            return f"❌ Error updating voice client data: {str(e)}"
    
    def create_interface(self) -> Tuple[gr.Blocks, gr.Textbox]:
        """Create the complete client dashboard interface"""
        with gr.Blocks(title="🎯 Comprehensive Client Information Portal", theme=gr.themes.Soft()) as demo:
            
            gr.Markdown("# 🎯 **COMPREHENSIVE CLIENT DASHBOARD**")
            gr.Markdown("### 📊 Complete client information, financial data, and communication history")
            
            # Search Panel
            user_id_input, search_button, perf_indicator = self.ui_builder.create_search_panel()
            
            # Main Content
            with gr.Row():
                with gr.Column(scale=3, elem_classes=["main-content"]):
                    with gr.Tabs():
                        with gr.TabItem("👤 Client Profile"):
                            vehicles_table, profile_md = self.ui_builder.create_profile_tab()
                        
                        with gr.TabItem("📊 Account Overview"):
                            account_outputs = self.ui_builder.create_account_tab()
                        
                        with gr.TabItem("📋 Client Contracts"):
                            contracts_table, contracts_summary = self.ui_builder.create_contracts_tab()
                        
                        with gr.TabItem("💳 Account Statement"):
                            statement_table, statement_summary = self.ui_builder.create_statement_tab()
                        
                        with gr.TabItem("🏦 Banking Details"):
                            banking_details = self.ui_builder.create_banking_tab()
                        
                        with gr.TabItem("💰 Payment History"):
                            payment_outputs = self.ui_builder.create_payment_tab()
                        
                        with gr.TabItem("🏦 Debit Mandates"):
                            mandates_table, mandates_summary = self.ui_builder.create_mandates_tab()
                        
                        with gr.TabItem("📋 Notes & Letters"):
                            notes_outputs = self.ui_builder.create_notes_tab()
            
            # Voice Chat Block - UPDATED IMPLEMENTATION WITH CLIENT DATA CACHING
            with gr.Column(elem_classes=["voice-chat-floating"]):
                # Create workflow factory that takes client_data (not user_id)
                def workflow_factory(client_data: dict):
                    """Create call center agent from pre-loaded client data."""
                    try:
                        if client_data:
                            from src.Agents.graph_call_center_agent import create_call_center_agent
                            from src.Agents.call_center_agent.call_scripts import ScriptType
                            from langchain_ollama import ChatOllama
                            
                            llm = ChatOllama(model="qwen2.5:14b-instruct", temperature=0, num_ctx=32000)
                            
                            return create_call_center_agent(
                                model=llm,
                                client_data=client_data,
                                script_type=ScriptType.RATIO_1_INFLOW.value,
                                config=CONFIG
                            )
                        return None
                    except Exception as e:
                        print(f"Error creating workflow: {e}")
                        return None
                
                # Create voice handler with workflow factory
                voice_handler = VoiceInteractionHandler(CONFIG, workflow_factory)
                
                # Create voice chat block with client data loader
                voice_chat_block = create_voice_chat_block(
                    voice_handler=voice_handler, 
                    workflow=None, 
                    workflow_factory=workflow_factory,
                    client_data_loader=self.load_client_data_for_voice
                )
                
            # Define all outputs
            outputs = [
                vehicles_table, profile_md,  # Profile (2)
                *account_outputs,  # Account (7)
                contracts_table, contracts_summary,  # Contracts (2)
                statement_table, statement_summary,  # Statement (2)
                banking_details,  # Banking (1)
                *payment_outputs,  # Payment History (3)
                mandates_table, mandates_summary,  # Mandates (2)
                *notes_outputs  # Notes & Letters (3)
            ]
            
            # Function to update voice chat user ID and load client data
            def update_voice_chat_user_id_and_data(user_id: str) -> Tuple[str, str]:
                """Update the voice chat user ID and load client data."""
                # Update user ID state
                user_id = user_id.strip() if user_id else ""
                
                # Update client data in voice chat
                if user_id:
                    status = self.update_voice_client_data(user_id, voice_chat_block)
                else:
                    status = "❌ No client selected"
                
                return user_id, status
            
            # Event handlers
            search_button.click(
                fn=self.timed_update,
                inputs=[user_id_input],
                outputs=outputs + [perf_indicator]
            )
            
            user_id_input.submit(
                fn=self.update_all_data,
                inputs=[user_id_input],
                outputs=outputs
            )
            
            user_id_input.change(
                fn=self.update_all_data,
                inputs=[user_id_input],
                outputs=outputs
            )
            
            # CRITICAL: Update voice chat when user_id_input changes
            user_id_input.change(
                fn=update_voice_chat_user_id_and_data,
                inputs=[user_id_input],
                outputs=[voice_chat_block.current_user_id, voice_chat_block.client_status_display]
            )
            
            # Also update on search
            search_button.click(
                fn=update_voice_chat_user_id_and_data,
                inputs=[user_id_input],
                outputs=[voice_chat_block.current_user_id, voice_chat_block.client_status_display]
            )
            
            # Initialize voice chat with initial client data
            demo.load(
                fn=update_voice_chat_user_id_and_data,
                inputs=[user_id_input],
                outputs=[voice_chat_block.current_user_id, voice_chat_block.client_status_display]
            )
        
        return demo, user_id_input


def create_client_details_interface(initial_client_id: str = "28173") -> Tuple[gr.Blocks, gr.Textbox]:
    """Create comprehensive client dashboard with all display functions"""
    dashboard = ClientDashboard(initial_client_id)
    return dashboard.create_interface()


# Usage example
# if __name__ == "__main__":
#     demo, user_input = create_client_details_interface()
#     demo.launch()