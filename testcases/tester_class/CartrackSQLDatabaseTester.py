# Import necessary libraries
import json
from IPython.display import display, HTML
import pandas as pd
from datetime import datetime
import time

class CartrackSQLDatabaseTester:
    """Interactive test utility for CartrackSQLDatabase class in Jupyter notebooks with color formatting."""
    
    # Color constants
    COLORS = {
        'SUCCESS': '\033[92m',  # Green
        'ERROR': '\033[91m',    # Red
        'WARNING': '\033[93m',  # Yellow
        'INFO': '\033[94m',     # Blue
        'HEADER': '\033[95m',   # Purple
        'BOLD': '\033[1m',      # Bold
        'UNDERLINE': '\033[4m', # Underline
        'END': '\033[0m'        # Reset
    }
    
    def __init__(self, db_instance):
        """Initialize with database instance."""
        self.db = db_instance
        self.test_results = []
    
    def colored(self, text, color_name):
        """
        Return text with color formatting.
        
        Args:
            text: The text to colorize
            color_name: The color name from COLORS dictionary
        """
        if color_name not in self.COLORS:
            return text
        return f"{self.COLORS[color_name]}{text}{self.COLORS['END']}"
    
    def print_header(self, text, width=80):
        """Print a formatted header with color."""
        padding = max(2, (width - len(text) - 2) // 2)
        stars = '*' * padding
        header = f"{stars} {text} {stars}"
        print(f"\n{self.colored(header, 'HEADER')}")
    
    def print_section(self, text):
        """Print a section title with color."""
        print(f"\n{self.colored('=== ' + text + ' ===', 'BOLD')}")
    
    def print_json(self, json_data, title=None):
        """
        Pretty print JSON data or convert to DataFrame if possible.
        
        Args:
            json_data: Dictionary or list of dictionaries to display
            title: Optional title for the output
        """
        if title:
            self.print_section(title)
            
        if json_data is None:
            print(self.colored("\n*** NO DATA RETURNED ***\n", 'WARNING'))
            return 0
        
        if not isinstance(json_data, list):
            json_data = [json_data]
            
        if not json_data or not json_data[0]:
            print(self.colored("\n*** EMPTY DATA ***\n", 'WARNING'))
            return 0
        
        # Try to convert to DataFrame for better display
        try:
            df = pd.DataFrame(json_data)
            display(df)
            # Save row count for reporting
            row_count = len(df)
            col_count = len(df.columns)
            print(self.colored(f"Rows: {row_count}, Columns: {col_count}", 'INFO'))
            return row_count
        except:
            # Fall back to regular printing if DataFrame conversion fails
            print(self.colored("*" * 75, 'INFO'))
            for i, result in enumerate(json_data):
                print(self.colored(f"{'-'*5} Record {i} {'-'*50}", 'BOLD'))
                for (k, v) in result.items():
                    # Truncate very long values
                    if isinstance(v, str) and len(v) > 100:
                        v = f"{v[:97]}..."
                    print(f"{self.colored(k, 'BOLD')}: {v}")
            print(self.colored("*" * 75, 'INFO'))
            return len(json_data)
    
    def test_method(self, method_name, *args, **kwargs):
        """
        Test a specific method of the CartrackSQLDatabase class.
        
        Args:
            method_name: Name of the method to test
            *args: Arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method
        """
        # Get the method from the database instance
        method = getattr(self.db, method_name)
        
        # Display header
        self.print_header(f"TESTING {method_name}")
        
        # Format the arguments for display
        args_str = ', '.join(repr(arg) for arg in args)
        kwargs_str = ', '.join(f"{k}={repr(v)}" for k, v in kwargs.items())
        all_args = ', '.join(filter(None, [args_str, kwargs_str]))
        
        print(f"Calling: {self.colored(method_name, 'BOLD')}({self.colored(all_args, 'INFO')})")
        
        # Execute the method with provided arguments
        start_time = time.time()
        try:
            result = method(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Print the execution time
            print(f"Execution time: {self.colored(f'{execution_time:.4f} seconds', 'INFO')}")
            
            # Print the result
            row_count = self.print_json(result, f"Results from {method_name}")
            
            # Save test result
            self.test_results.append({
                'method': method_name,
                'args': all_args,
                'status': 'SUCCESS',
                'rows': row_count,
                'time': execution_time,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            print(self.colored("✓ Test completed successfully", 'SUCCESS'))
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Print the error
            print(f"\n{self.colored('⚠️ ERROR in ' + method_name + ': ' + str(e), 'ERROR')}\n")
            
            # Save test result
            self.test_results.append({
                'method': method_name,
                'args': all_args,
                'status': 'ERROR',
                'error': str(e),
                'time': execution_time,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return None
    
    def run_multiple_tests(self, user_id, methods=None):
        """
        Run multiple test methods for a given user_id.
        
        Args:
            user_id: The user ID to test with
            methods: List of method names to test, or None for default test
        """
        self.print_header(f"RUNNING MULTIPLE TESTS FOR USER ID: {user_id}")
        
        if methods is None:
            # Default is to run only the debit order mandate test
            self.test_method("get_client_debit_order_mandate", user_id)
            return
        
        for method in methods:
            self.test_method(method, user_id)
            
        self.display_test_summary(f"Tests for user {user_id}")
    
    def run_all_tests(self, user_id):
        """
        Run all available tests for a given user_id.
        
        Args:
            user_id: The user ID to test with
        """
        self.print_header(f"RUNNING ALL TESTS FOR USER ID: {user_id}")
        
        # Define all available read test methods with their arguments
        read_test_methods = [
            "get_billing_age_analysis",
            "get_debtor_billing_age_analysis",
            "get_debtor_info",
            "get_client_vehicle_info",
            "get_client_contract_info",
            "get_sp_debtors_age",
            "get_client_call_verification_token",
            "get_client_debit_order_mandate",
            "get_sp_client_debtor_notes",
            "get_sp_debtors_account_statement",
            "get_sp_debtors_client_info",
            "get_sp_debtor_client_positions",
            "load_list_debtor_letters",
            "get_client_information_for_verification",
            # New methods added
            "get_sp_client_information",
            "get_sp_get_client_account_state",
            "get_sp_debtors_account_info",
            "get_client_arrangements",
            "get_payment_history",
            "get_monthly_subscription_amount",
            "validate_next_of_kin_contact",
            "validate_next_of_kin_emergency_contact"
        ]
        
        for method in read_test_methods:
            if method == "get_billing_age_analysis":
                self.test_method(method, "WHERE balance_total > 0 AND client_type='Individual' LIMIT 2")
            else:
                self.test_method(method, user_id)
                
        self.display_test_summary(f"All read tests for user {user_id}")
    
    def run_read_tests(self, user_id):
        """
        Run all read tests for a given user_id.
        
        Args:
            user_id: The user ID to test with
        """
        self.print_header(f"RUNNING READ TESTS FOR USER ID: {user_id}")
        
        # Define read test methods
        read_methods = [
            "get_debtor_info",
            "get_client_vehicle_info",
            "get_client_contract_info",
            "get_sp_debtors_age",
            "get_client_debit_order_mandate",
            "get_sp_client_debtor_notes",
            "get_sp_debtors_account_statement",
            "get_sp_debtors_client_info",
            "get_client_information_for_verification",
            "get_sp_client_information",
            "get_sp_get_client_account_state",
            "get_sp_debtors_account_info",
            "get_client_arrangements",
            "get_payment_history",
            "get_monthly_subscription_amount"
        ]
        
        for method in read_methods:
            self.test_method(method, user_id)
            
        self.display_test_summary(f"Read tests for user {user_id}")
    
    def test_create_payment_link(self, user_id, amount=100.00):
        """Test creating a payment link for SMS."""
        self.print_header(f"TESTING PAYMENT LINK CREATION")
        return self.test_method("create_payment_link", user_id, amount)
    
    def test_add_client_note(self, user_id, note_text="Test note from database tester"):
        """Test adding a client note."""
        self.print_header(f"TESTING CLIENT NOTE ADDITION")
        return self.test_method("add_client_note", user_id, note_text)
    
    def run_write_tests_simulation(self, user_id):
        """
        Run simulations of write operations without actually writing data.
        This is safer for testing database interactions.
        
        Args:
            user_id: The user ID to test with
        """
        self.print_header(f"RUNNING WRITE TEST SIMULATIONS FOR USER ID: {user_id}")
        
        # Define simulated write test scenarios
        write_simulations = [
            {
                "title": "Update Bank Account Simulation",
                "description": "Simulates updating bank account details",
                "params": {
                    "user_id": user_id,
                    "bank_name": "TEST BANK",
                    "account_number": "1234567890",
                    "branch_code": "12345",
                    "account_type": "SAVINGS"
                },
                "method": "update_user_bank_account"
            },
            {
                "title": "Update Individual Contact Simulation",
                "description": "Simulates updating contact information",
                "params": {
                    "user_id": user_id,
                    "cell_number": "0712345678",
                    "email": "test@example.com"
                },
                "method": "update_individual_contact"
            },
            {
                "title": "Update Address Simulation",
                "description": "Simulates updating address information",
                "params": {
                    "user_id": user_id,
                    "address_line_1": "123 Test Street",
                    "suburb": "Test Suburb",
                    "city": "Test City",
                    "postal_code": "12345"
                },
                "method": "update_individual_address"
            },
            {
                "title": "Create Arrangement Simulation",
                "description": "Simulates creating a payment arrangement",
                "params": {
                    "user_id": user_id,
                    "amount": 100.00,
                    "date": datetime.now().strftime('%Y-%m-%d'),
                    "payment_method": "EFT",
                    "reference": "TEST PAYMENT"
                },
                "method": "update_arrangement"
            },
            {
                "title": "Create Referral Simulation",
                "description": "Simulates creating a referral",
                "params": {
                    "user_id": user_id,
                    "referrer_name": "Test Referrer",
                    "referrer_number": "0712345678",
                    "notes": "Test referral note"
                },
                "method": "create_referral"
            },
            {
                "title": "Create Helpdesk Ticket Simulation",
                "description": "Simulates creating a helpdesk ticket",
                "params": {
                    "user_id": user_id,
                    "title": "Test Cancellation Request",
                    "description": "This is a test cancellation request",
                    "priority": "Medium"
                },
                "method": "create_helpdesk_ticket"
            },
            {
                "title": "Save Call Disposition Simulation",
                "description": "Simulates saving a call disposition",
                "params": {
                    "user_id": user_id,
                    "disposition_code": "TEST",
                    "disposition_notes": "Test disposition note"
                },
                "method": "save_call_disposition"
            }
        ]
        
        # Print simulations without executing them
        for sim in write_simulations:
            self.print_section(sim["title"])
            print(f"{self.colored('Description:', 'BOLD')} {sim['description']}")
            print(f"{self.colored('Method:', 'INFO')} {sim['method']}")
            print(f"{self.colored('Parameters:', 'INFO')}")
            
            for k, v in sim["params"].items():
                print(f"  {self.colored(k, 'BOLD')}: {v}")
                
            print(f"\n{self.colored('Note:', 'WARNING')} This operation would modify the database and is not executed in simulation mode.")
    
    def test_run_all_functions(self, user_id, client_id=None):
        """
        Run all available functions for a given user_id, organized by category.
        
        Args:
            user_id: The user ID to test with
            client_id: Optional client ID if different from user_id
        """
        if client_id is None:
            client_id = user_id
            
        self.print_header(f"COMPREHENSIVE TESTING FOR USER ID: {user_id} / CLIENT ID: {client_id}")
        
        # Group methods by category for organized testing
        test_categories = {
            "Client Verification": [
                "get_debtor_info",
                "get_client_information_for_verification",
                "get_sp_client_information",
                "get_client_vehicle_info"
            ],
            "Account Information": [
                "get_sp_debtors_age",
                "get_sp_get_client_account_state",
                "get_sp_debtors_account_info",
                "get_sp_debtors_account_statement",
                "get_monthly_subscription_amount",
                "get_sp_debtors_client_info"
            ],
            "Contract & Mandate": [
                "get_client_contract_info",
                "get_client_debit_order_mandate",
                "get_client_call_verification_token"
            ],
            "Payment History": [
                "get_client_arrangements",
                "get_payment_history"
            ],
            "Notes & Communications": [
                "get_sp_client_debtor_notes",
                "load_list_debtor_letters",
                "validate_next_of_kin_contact",
                "validate_next_of_kin_emergency_contact"
            ]
        }
        
        # Run tests by category
        for category, methods in test_categories.items():
            self.print_section(category)
            for method in methods:
                # Determine if method needs user_id or client_id
                if "client_" in method and "client_information" not in method:
                    self.test_method(method, client_id)
                else:
                    self.test_method(method, user_id)
                    
        self.display_test_summary(f"Comprehensive tests for {user_id}")
    
    def display_test_summary(self, title="TEST SUMMARY"):
        """Display a summary of all tests run."""
        if not self.test_results:
            print(self.colored("No tests have been run yet.", 'WARNING'))
            return
        
        self.print_header(title)
        
        df = pd.DataFrame(self.test_results)
        
        # Create HTML with embedded styles for better visualization in Jupyter
        def style_dataframe(df):
            # Create a styled dataframe with better colors
            styler = df.style.apply(
                lambda x: [
                    'background-color: rgba(60, 179, 113, 0.2); color: black' 
                    if x['status'] == 'SUCCESS' 
                    else 'background-color: rgba(255, 99, 71, 0.2); color: black'
                    for _ in x
                ],
                axis=1
            )
            
            # Add more styling for specific columns
            styler = styler.apply(
                lambda x: [
                    'font-weight: bold; border-right: 1px solid #ddd' if i == 0 
                    else 'font-weight: bold; color: green' if i == 2 and v == 'SUCCESS'
                    else 'font-weight: bold; color: red' if i == 2 and v == 'ERROR'
                    else 'font-style: italic; color: #666' if i == 1
                    else '' 
                    for i, v in enumerate(x)
                ],
                axis=1
            )
            
            # Set table styles for better borders and spacing
            styler = styler.set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#f2f2f2'), 
                                             ('color', 'black'),
                                             ('font-weight', 'bold'),
                                             ('border', '1px solid #ddd'),
                                             ('padding', '8px')]},
                {'selector': 'td', 'props': [('border', '1px solid #ddd'),
                                             ('padding', '8px')]},
                {'selector': 'tr:hover', 'props': [('background-color', '#f5f5f5')]},
                {'selector': 'table', 'props': [('border-collapse', 'collapse'), 
                                               ('width', '100%'),
                                               ('margin', '8px 0')]},
            ])
            
            return styler
        
        # Apply improved styling
        styled_df = style_dataframe(df)
        
        display(styled_df)
        
        # Print overall statistics
        success_count = df[df['status'] == 'SUCCESS'].shape[0]
        error_count = df[df['status'] == 'ERROR'].shape[0]
        total_time = df['time'].sum()
        
        print(f"\n{self.colored('Total tests:', 'BOLD')} {len(df)}")
        print(f"{self.colored('Successful:', 'SUCCESS')} {success_count}")
        print(f"{self.colored('Failed:', 'ERROR')} {error_count}")
        print(f"{self.colored('Total execution time:', 'BOLD')} {total_time:.2f} seconds")
        
        # If there were errors, print them out for easy review
        if error_count > 0:
            self.print_section("ERRORS")
            error_df = df[df['status'] == 'ERROR'][['method', 'args', 'error', 'time']]
            
            # Format the error data as an HTML table for better readability
            from IPython.display import HTML
            
            html = "<table style='width:100%; border-collapse:collapse; margin:10px 0'>"
            html += "<tr style='background-color:#f2f2f2'><th style='border:1px solid #ddd; padding:8px; text-align:left'>Method</th>"
            html += "<th style='border:1px solid #ddd; padding:8px; text-align:left'>Arguments</th>"
            html += "<th style='border:1px solid #ddd; padding:8px; text-align:left'>Error</th>"
            html += "<th style='border:1px solid #ddd; padding:8px; text-align:right'>Time (s)</th></tr>"
            
            for idx, row in error_df.iterrows():
                html += "<tr style='background-color:rgba(255,99,71,0.1)'>"
                html += f"<td style='border:1px solid #ddd; padding:8px; font-weight:bold'>{row['method']}</td>"
                html += f"<td style='border:1px solid #ddd; padding:8px; font-style:italic'>{row['args']}</td>"
                html += f"<td style='border:1px solid #ddd; padding:8px; color:#d32f2f'>{row['error']}</td>"
                html += f"<td style='border:1px solid #ddd; padding:8px; text-align:right'>{row['time']:.4f}</td>"
                html += "</tr>"
            
            html += "</table>"
            display(HTML(html))
            
            # Also print to standard output for easier copy-paste
            for idx, row in error_df.iterrows():
                print(f"{self.colored(row['method'], 'BOLD')}({row['args']}): {self.colored(row['error'], 'ERROR')}")
                
    def create_method_documentation(self):
        """Generate documentation of all available methods in the database class."""
        self.print_header("DATABASE METHOD DOCUMENTATION")
        
        # Get all methods from the database class
        methods = [method for method in dir(self.db) 
                 if callable(getattr(self.db, method)) and not method.startswith('_')]
        
        # Group methods by their prefix for better organization
        method_groups = {}
        for method in methods:
            # Extract prefix (get_, update_, create_, etc.)
            parts = method.split('_')
            if len(parts) > 1:
                prefix = parts[0]
                if prefix not in method_groups:
                    method_groups[prefix] = []
                method_groups[prefix].append(method)
            else:
                if 'other' not in method_groups:
                    method_groups['other'] = []
                method_groups['other'].append(method)
        
        # Create documentation table
        doc_rows = []
        
        for group, group_methods in sorted(method_groups.items()):
            # Add a group header
            doc_rows.append({
                'method': f'===== {group.upper()} METHODS =====',
                'description': '',
                'parameters': '',
                'returns': ''
            })
            
            # Add methods in this group
            for method in sorted(group_methods):
                func = getattr(self.db, method)
                doc = func.__doc__ or "No documentation available"
                # Extract parameters from method signature
                import inspect
                sig = inspect.signature(func)
                params = []
                for name, param in sig.parameters.items():
                    if name != 'self':
                        if param.default != inspect.Parameter.empty:
                            params.append(f"{name}={param.default}")
                        else:
                            params.append(name)
                
                # Extract return type if annotated
                return_type = "Unknown"
                if hasattr(func, "__annotations__") and "return" in func.__annotations__:
                    return_type = str(func.__annotations__["return"]).replace("typing.", "")
                
                doc_rows.append({
                    'method': method,
                    'description': doc.strip(),
                    'parameters': ", ".join(params),
                    'returns': return_type
                })
        
        # Display the documentation
        doc_df = pd.DataFrame(doc_rows)
        
        # Create HTML for better display
        html = "<table style='width:100%; border-collapse:collapse; margin:10px 0'>"
        html += "<tr style='background-color:#f2f2f2'>"
        html += "<th style='border:1px solid #ddd; padding:8px; text-align:left; width:25%'>Method</th>"
        html += "<th style='border:1px solid #ddd; padding:8px; text-align:left; width:40%'>Description</th>"
        html += "<th style='border:1px solid #ddd; padding:8px; text-align:left; width:20%'>Parameters</th>"
        html += "<th style='border:1px solid #ddd; padding:8px; text-align:left; width:15%'>Returns</th>"
        html += "</tr>"
        
        for idx, row in doc_df.iterrows():
            # Check if this is a group header
            if row['method'].startswith('====='):
                html += f"<tr style='background-color:#e6f3ff'>"
                html += f"<td colspan='4' style='border:1px solid #ddd; padding:8px; font-weight:bold; text-align:center'>{row['method'].replace('=====', '').strip()}</td>"
                html += "</tr>"
            else:
                html += "<tr>"
                html += f"<td style='border:1px solid #ddd; padding:8px; font-weight:bold'>{row['method']}</td>"
                html += f"<td style='border:1px solid #ddd; padding:8px'>{row['description']}</td>"
                html += f"<td style='border:1px solid #ddd; padding:8px; font-style:italic'>{row['parameters']}</td>"
                html += f"<td style='border:1px solid #ddd; padding:8px'>{row['returns']}</td>"
                html += "</tr>"
        
        html += "</table>"
        display(HTML(html))