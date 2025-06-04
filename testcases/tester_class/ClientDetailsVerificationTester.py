import time
import json
import re
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage

# ANSI color codes for terminal output
COLORS = {
    "END": "\033[0m",
    "HEADER": "\033[95m",
    "INFO": "\033[94m",
    "SUCCESS": "\033[92m",
    "WARNING": "\033[93m",
    "ERROR": "\033[91m",
    "BOLD": "\033[1m"
}

def is_empty_value(value):
    """
    Check if a value is empty or None.
    
    Args:
        value: The value to check
        
    Returns:
        True if the value is None, empty string, or just whitespace
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False

class ClientDetailsVerificationTester:
    """A test harness for evaluating the client details verification tool against expected classifications."""
    
    def __init__(self, verify_client_details: Any, default_required_matches=3, verbose=False):
        """
        Initialize the tester with default settings.
        
        Args:
            verify_client_details: The function that verifies client details
            default_required_matches (int): Number of matches required for verification
            verbose (bool): Flag to show detailed output
        """
        self.verify_client_details = verify_client_details
        self.default_required_matches = default_required_matches
        self.test_results = []
        self.verbose = verbose  # Control level of detail in output
        self.sessions = {}  # Track sessions for each test case
    
    def run_test(self, client_details, messages, expected_result=None, required_matches=None, test_number=None, use_session=True):
        """
        Run a single test case for the client details verification tool.
        
        Args:
            client_details (dict): Dictionary containing the client's records
            messages (list): List of message objects (HumanMessage or AIMessage)
            expected_result (str, optional): Expected classification
            required_matches (int, optional): Custom required match count
            test_number (int, optional): Test case number for display
            use_session (bool): Whether to maintain session state between test runs
            
        Returns:
            dict: The verification result object
        """
        test_required_matches = required_matches if required_matches is not None else self.default_required_matches
        
        # Record test case details
        test_case = {
            "client_details": client_details,
            "messages": [{"role": "assistant" if isinstance(m, AIMessage) else "human", 
                         "content": m.content} for m in messages],
            "required_matches": test_required_matches
        }
        
        # Start timer
        start_time = time.time()
        
        # Get session state if available and enabled
        session_state = None
        if use_session and test_number is not None and test_number in self.sessions:
            session_state = self.sessions[test_number]
        
        try:
            # Run verification with session state if available
            invoke_params = {
                'client_details': client_details,
                'messages': messages,
                'required_match_count': test_required_matches,
                'max_failed_attempts': 3
            }
            
            if session_state:
                invoke_params['session_state'] = session_state
                
            result = self.verify_client_details.invoke(invoke_params)
            
            # Store session state for future use
            if test_number is not None and 'session_state' in result:
                self.sessions[test_number] = result['session_state']
                
        except Exception as e:
            print(f"{COLORS['ERROR']}Error invoking verify_client_details: {e}{COLORS['END']}")
            result = {
                "classification": "ERROR",
                "matched_fields": [],
                "verification_attempts": 0,
                "error": str(e)
            }
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Check if result matches expected result (if provided)
        passed = (expected_result is None or result['classification'] == expected_result)
        
        # Store test result with adapted fields structure for compatibility
        test_result = {
            "test_case": test_case,
            "result": {
                "classification": result['classification'],
                "matched_fields": result.get('matched_fields', []),
                "unmatched_fields": [],  # Empty since we don't track this in optimized version
                "verification_attempts": result.get('verification_attempts', 0),
                "failed_attempts": 0,  # Not tracked in optimized version
                "extracted_fields": result.get('extracted_fields', {}),  # Now tracked in optimized version
                "reasoning": result.get('error', 'No reasoning provided in optimized version')
            },
            "metrics": {},  # Not tracked in optimized version
            "execution_time": execution_time,
            "passed": passed,
            "expected": expected_result,
            "test_number": test_number
        }
        self.test_results.append(test_result)
        
        return result
    
    def _print_test_result_optimized(self, test_result, test_number=None):
        """
        Print a simplified, formatted result for a single test case.
        
        Args:
            test_result (dict): The test result to print
            test_number (int, optional): Number of the test case
        """
        # Set a consistent width for all boxes
        width = 70
        
        client_name = test_result['test_case']['client_details'].get('full_name', 'Unknown Client')
        
        # Determine test result status for the header
        passed = test_result['passed']
        status_text = "PASS ✓" if passed else "FAIL ✗"
        
        # Create a horizontal border line
        border_line = "─" * width
        
        # Print separation line between test cases
        print(f"\n{COLORS['HEADER']}{'═' * (width + 2)}{COLORS['END']}")
        
        # ===== HEADER BOX =====
        test_num_prefix = f"[{test_number}] " if test_number is not None else ""
        header_text = f"TEST CASE: {test_num_prefix}{client_name}"
        status_color = COLORS['SUCCESS'] if passed else COLORS['ERROR']
        
        # Print the header box
        print(f"{COLORS['HEADER']}┌{border_line}┐{COLORS['END']}")
        
        # Calculate status position to ensure right alignment
        status_position = width - len(status_text) - 1
        header_line = f"{header_text}{' ' * (status_position - len(header_text))}{status_text}"
        
        # Print header with colored status
        plain_header = header_line[:status_position]
        print(f"{COLORS['HEADER']}│{COLORS['END']} {plain_header}{status_color}{status_text}{COLORS['END']} {COLORS['HEADER']}│{COLORS['END']}")
        print(f"{COLORS['HEADER']}└{border_line}┘{COLORS['END']}")
        
        # ----- CONVERSATION BOX -----
        print(f"\n{COLORS['BOLD']}CONVERSATION:{COLORS['END']}")
        print(f"┌{border_line}┐")
        
        for idx, msg in enumerate(test_result['test_case']['messages']):
            # Set role colors
            role_color = COLORS['INFO'] if msg['role'] == 'assistant' else COLORS['WARNING']
            role_display = "Agent" if msg['role'] == 'assistant' else "Client"
            
            # Print role header with color
            print(f"│ {role_color}{role_display}:{COLORS['END']}{' ' * (width - len(role_display) - 3)}│")
            
            # Wrap message content
            content_lines = []
            remaining = msg['content']
            max_content_width = width - 4  # Space for "│  " and " │"
            
            while remaining:
                if len(remaining) <= max_content_width:
                    content_lines.append(remaining)
                    break
                
                split_pos = remaining[:max_content_width].rfind(' ')
                if split_pos == -1:
                    split_pos = max_content_width
                    
                content_lines.append(remaining[:split_pos])
                remaining = remaining[split_pos:].lstrip()
            
            # Print each line of content with color
            for line in content_lines:
                padding = max_content_width - len(line)
                print(f"│  {role_color}{line}{COLORS['END']}{' ' * padding} │")
            
            # Add separator between messages unless it's the last message
            if idx < len(test_result['test_case']['messages']) - 1:
                print(f"│{' ' * width}│")
        
        print(f"└{border_line}┘")
        
        # ----- VERIFICATION RESULTS BOX -----
        result_color = {
            "VERIFIED": COLORS['SUCCESS'],
            "INSUFFICIENT_INFO": COLORS['WARNING'],
            "VERIFICATION_FAILED": COLORS['ERROR'],
            "ERROR": COLORS['ERROR']
        }.get(test_result['result']['classification'], COLORS['INFO'])
        
        expected_color = {
            "VERIFIED": COLORS['SUCCESS'],
            "INSUFFICIENT_INFO": COLORS['WARNING'],
            "VERIFICATION_FAILED": COLORS['ERROR']
        }.get(test_result['expected'], COLORS['INFO'])
        
        print(f"\n{COLORS['BOLD']}VERIFICATION RESULTS:{COLORS['END']}")
        print(f"┌{border_line}┐")
        
        # First line of results
        model_class = test_result['result']['classification']
        match_status = "Match" if passed else "Mismatch"
        match_color = COLORS['SUCCESS'] if passed else COLORS['ERROR']
        
        cls_section = f"Model Classification: {model_class}"
        cls_section_with_color = f"Model Classification: {result_color}{model_class}{COLORS['END']}"
        
        # Calculate padding to ensure alignment
        padding = width - len(cls_section) - len(" │  │") - len(match_status)
        status_section = f"{' ' * padding} │ {match_color}{match_status}{COLORS['END']}"
        
        print(f"│ {cls_section_with_color}{status_section} │")
        
        # Second line of results
        expected = test_result['expected']
        expected_section = f"Expected Result:      {expected}"
        expected_section_with_color = f"Expected Result:      {expected_color}{expected}{COLORS['END']}"
        
        # Keep the right side aligned with the previous line
        print(f"│ {expected_section_with_color}{' ' * (width - len(expected_section) - 2)}│")
        
        # Third line with attempts count
        verification_attempts = test_result['result']['verification_attempts']
        verification_section = f"Verification Attempts: {verification_attempts}"
        
        print(f"│ {verification_section}{' ' * (width - len(verification_section) - 2)}│")
        print(f"└{border_line}┘")
        
        # ----- EXTRACTED FIELDS SECTION -----
        print(f"\n{COLORS['BOLD']}EXTRACTED FIELDS:{COLORS['END']}")
        extracted_fields = test_result['result'].get('extracted_fields', {})
        
        # Filter out empty values
        non_empty_extractions = {k: v for k, v in extracted_fields.items() if not is_empty_value(v)}
        
        if non_empty_extractions:
            for field, value in non_empty_extractions.items():
                db_value = test_result['test_case']['client_details'].get(field, "N/A")
                is_matched = field in test_result['result']['matched_fields']
                status_symbol = "✓" if is_matched else "✗"
                status_color = COLORS['SUCCESS'] if is_matched else COLORS['ERROR'] 
                print(f"  {status_color}{status_symbol}{COLORS['END']} {field}: Extracted '{COLORS['BOLD']}{value}{COLORS['END']}', Database '{db_value}'")
        else:
            print(f"  {COLORS['WARNING']}No fields extracted{COLORS['END']}")
            
        # ----- MATCHED FIELDS

    def _print_summary_optimized(self):
        """Print a concise summary of all test results."""
        total = len(self.test_results)
        if total == 0:
            print(f"\n{COLORS['WARNING']}No tests were run.{COLORS['END']}")
            return
            
        passed = sum(1 for r in self.test_results if r['passed'])
        failed = total - passed
        
        # Calculate classification statistics
        classification_counts = {}
        for r in self.test_results:
            cls = r['result']['classification']
            classification_counts[cls] = classification_counts.get(cls, 0) + 1
        
        expected_counts = {}
        for r in self.test_results:
            if r['expected']:
                cls = r['expected']
                expected_counts[cls] = expected_counts.get(cls, 0) + 1
        
        # Define consistent box width for summary
        width = 70
        border_line = "─" * width
        
        # Print summary header with box
        print(f"\n{COLORS['HEADER']}{'═' * (width + 2)}{COLORS['END']}")
        print(f"{COLORS['HEADER']}┌{border_line}┐{COLORS['END']}")
        header_text = f"TEST SUMMARY ({total} tests)"
        print(f"{COLORS['HEADER']}│{COLORS['END']} {COLORS['BOLD']}{header_text}{COLORS['END']}{' ' * (width - len(header_text) - 1)}{COLORS['HEADER']}│{COLORS['END']}")
        print(f"{COLORS['HEADER']}└{border_line}┘{COLORS['END']}")
        
        # Print overall results section
        print(f"\n{COLORS['BOLD']}Overall Results:{COLORS['END']}")
        print(f"  {'Metric':<25} {'Count':<8} {'Percentage':<12}")
        print(f"  {'-'*25} {'-'*8} {'-'*12}")
        
        # Expected results with highlighted numbers
        result_color = COLORS['SUCCESS'] if passed == total else COLORS['WARNING']
        print(f"  {'Tests Passed':<25} {COLORS['SUCCESS']}{COLORS['BOLD']}{passed}{COLORS['END']:<8} {result_color}{passed/total*100:.1f}%{COLORS['END']}")
        print(f"  {'Tests Failed':<25} {COLORS['ERROR']}{COLORS['BOLD']}{failed}{COLORS['END']:<8} {COLORS['ERROR']}{failed/total*100:.1f}%{COLORS['END']}")
        
        # Increase first column width by 1.2x
        first_col_width = int(18 * 1.2)
        
        # Classification distribution
        print(f"\n{COLORS['BOLD']}Classification Distribution:{COLORS['END']}")
        
        # Color mapping
        result_colors = {
            "VERIFIED": COLORS['SUCCESS'],
            "INSUFFICIENT_INFO": COLORS['WARNING'],
            "VERIFICATION_FAILED": COLORS['ERROR'],
            "ERROR": COLORS['ERROR']
        }
        
        # Create a side-by-side comparison table
        print(f"  {'Classification':<{first_col_width}} {'Model Predicted':<15} {'Expected':<15}")
        print(f"  {'-'*first_col_width} {'-'*15} {'-'*15}")
        
        # Get unique classification values for matrix
        all_classifications = set(list(classification_counts.keys()) + list(expected_counts.keys()))
        
        for cls in sorted(all_classifications):
            color = result_colors.get(cls, COLORS['INFO'])
            model_count = classification_counts.get(cls, 0)
            expected_count = expected_counts.get(cls, 0)
            
            model_pct = f"({model_count/total*100:.1f}%)" if model_count > 0 else ""
            expected_pct = f"({expected_count/total*100:.1f}%)" if expected_count > 0 else ""
            
            print(f"  {color}{cls:<{first_col_width}}{COLORS['END']} {model_count:<5} {model_pct:<9} {expected_count:<5} {expected_pct:<9}")
        
        # Print confusion matrix
        print(f"\n{COLORS['BOLD']}Confusion Matrix:{COLORS['END']}")
        print(f"  Actual (rows) vs. Predicted (columns)")
        
        # Get unique classification values for matrix
        classifications = ["VERIFIED", "INSUFFICIENT_INFO", "VERIFICATION_FAILED"]
        
        # Print header
        header = "  " + " " * first_col_width
        for cls in classifications:
            header += f" {cls[:5]:<10}"
        print(header)
        
        # Build confusion matrix
        confusion_matrix = {}
        for r in self.test_results:
            if r['expected'] is not None:
                key = (r['expected'], r['result']['classification'])
                confusion_matrix[key] = confusion_matrix.get(key, 0) + 1
        
        # Print matrix
        for actual in classifications:
            line = f"  {actual:<{first_col_width}}"
            for predicted in classifications:
                count = confusion_matrix.get((actual, predicted), 0)
                if actual == predicted:
                    # Matched classifications in green
                    line += f" {COLORS['SUCCESS']}{count:<10}{COLORS['END']}"
                elif count > 0:
                    # Wrong classifications in red
                    line += f" {COLORS['ERROR']}{count:<10}{COLORS['END']}"
                else:
                    # Zero counts without color
                    line += f" {count:<10}"
            print(line)
        
        # Print average execution time
        avg_time = sum(r['execution_time'] for r in self.test_results) / total
        print(f"\n{COLORS['INFO']}Average execution time: {avg_time:.3f} seconds{COLORS['END']}")

    def run_all_tests(self, test_cases):
        """
        Run all predefined test cases with optimized output.
        
        Args:
            test_cases (list): List of test case dictionaries
        """
        print(f"\n{COLORS['HEADER']}{COLORS['BOLD']}==== RUNNING {len(test_cases)} VERIFICATION TESTS ===={COLORS['END']}\n")
        
        # Run all test cases
        for i, case in enumerate(test_cases):
            result = self.run_test(
                client_details=case["client_details"],
                messages=case["messages"],
                expected_result=case["expected"],
                required_matches=case.get("required_matches"),
                test_number=i+1,
            )
            
            # Update the printed test result with test number
            self._print_test_result_optimized(self.test_results[-1], i+1)
        
        # Print optimized summary
        self._print_summary_optimized()
    
    def export_results(self, filename="client_details_verification_test_results.json"):
        """
        Export test results to a JSON file.
        
        Args:
            filename (str, optional): Name of the file to export results to
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            print(f"\n{COLORS['SUCCESS']}Test results exported to {COLORS['BOLD']}{filename}{COLORS['END']}")
        except Exception as e:
            print(f"\n{COLORS['ERROR']}Error exporting results: {e}{COLORS['END']}")
    
    def generate_detailed_report(self, filename="verification_test_report.md"):
        """
        Generate a simplified Markdown report of test results.
        
        Args:
            filename (str, optional): Name of the Markdown report file
        """
        try:
            with open(filename, 'w') as f:
                # Report Header
                f.write("# Client Details Verification Test Report\n\n")
                
                # Overall Summary
                total_tests = len(self.test_results)
                passed_tests = sum(1 for r in self.test_results if r['passed'])
                f.write(f"## Test Summary\n")
                f.write(f"- **Total Tests:** {total_tests}\n")
                f.write(f"- **Passed Tests:** {passed_tests}\n")
                f.write(f"- **Pass Rate:** {passed_tests/total_tests*100:.2f}%\n\n")
                
                # Detailed Test Results
                f.write("## Detailed Test Results\n\n")
                for i, result in enumerate(self.test_results, 1):
                    f.write(f"### Test Case {i}: {result['test_case']['client_details'].get('full_name', 'Unknown Client')}\n\n")
                    f.write(f"- **Classification:** `{result['result']['classification']}`\n")
                    f.write(f"- **Expected Result:** `{result['expected']}`\n")
                    f.write(f"- **Status:** {'✅ PASSED' if result['passed'] else '❌ FAILED'}\n")
                    f.write(f"- **Verification Attempts:** {result['result']['verification_attempts']}\n\n")
                    
                    # Matched Fields
                    f.write("#### Matched Fields\n")
                    if result['result']['matched_fields']:
                        f.write("- " + "\n- ".join(result['result']['matched_fields']) + "\n\n")
                    else:
                        f.write("- *No fields matched*\n\n")
                
                # Classification Distribution
                f.write("## Classification Distribution\n")
                classification_counts = {}
                for r in self.test_results:
                    cls = r['result']['classification']
                    classification_counts[cls] = classification_counts.get(cls, 0) + 1
                
                for cls, count in classification_counts.items():
                    f.write(f"- **{cls}:** {count} ({count/total_tests*100:.2f}%)\n")
            
            print(f"\n{COLORS['SUCCESS']}Simplified report generated at {COLORS['BOLD']}{filename}{COLORS['END']}")
        except Exception as e:
            print(f"\n{COLORS['ERROR']}Error generating report: {e}{COLORS['END']}")

#########################################################################################
# Test cases
#########################################################################################
# Comprehensive Client Details Verification Test Cases
test_cases = [
    # ========== VERIFIED CASES (10 cases) ==========
    # These cases remain unchanged as they don't depend on failure pattern detection
    
    # ID-Based Verification
    {
        "client_details": {
            "full_name": "John Smith",
            "username": "jsmith2023",
            "vehicle_registration": "ABC123GP",
            "vehicle_make": "Toyota",
            "vehicle_model": "Corolla",
            "vehicle_color": "Blue",
            "email": "john.smith@example.com",
            "id_number": "8305125392086"
        },
        "messages": [
            AIMessage(content="For security purposes, could you verify your ID number?"),
            HumanMessage(content="Sure, it's 8305125392086.")
        ],
        "expected": "VERIFIED"
    },
    
    # Multiple Field Verification
    {
        "client_details": {
            "full_name": "Sarah Johnson",
            "username": "sjohnson",
            "vehicle_registration": "XYZ789WC",
            "vehicle_make": "Honda",
            "vehicle_model": "Civic",
            "vehicle_color": "Silver",
            "email": "sarah.j@example.com",
            "passport_number": "7602155392083"
        },
        "messages": [
            AIMessage(content="Please confirm three of your account details."),
            HumanMessage(content="I have a silver Honda Civic with registration XYZ789WC, and my email is sarah.j@example.com.")
        ],
        "expected": "VERIFIED"
    },
    
    # Detailed Verification with Various Fields
    {
        "client_details": {
            "full_name": "Michael Brown",
            "username": "mbrown2022",
            "vehicle_registration": "DEF456EC",
            "vehicle_make": "BMW",
            "vehicle_model": "X5",
            "vehicle_color": "Black",
            "email": "michael.brown@example.net",
            "passport_number": "7905235392085"
        },
        "messages": [
            AIMessage(content="Verify your identity by confirming account details."),
            HumanMessage(content="My username is mbrown2022, I drive a black BMW X5, and my registration is DEF456EC.")
        ],
        "expected": "VERIFIED"
    },
    
    # Precise Field Matching
    {
        "client_details": {
            "full_name": "Emma Wilson",
            "username": "ewilson",
            "vehicle_registration": "GHI789MP",
            "vehicle_make": "Audi",
            "vehicle_model": "A4",
            "vehicle_color": "White",
            "email": "emma.wilson@example.org",
            "passport_number": "8812045392087"
        },
        "messages": [
            AIMessage(content="Please provide three specific account details."),
            HumanMessage(content="My username is ewilson, I have a white Audi A4, and my email is emma.wilson@example.org.")
        ],
        "expected": "VERIFIED"
    },
    
    # Complex Multi-Field Verification
    {
        "client_details": {
            "full_name": "David Lee",
            "username": "dlee2021",
            "vehicle_registration": "JKL012NW",
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Golf",
            "vehicle_color": "Red",
            "email": "david.lee@example.com",
            "passport_number": "8107075392081"
        },
        "messages": [
            AIMessage(content="Confirm your identity with multiple details."),
            HumanMessage(content="My registration is JKL012NW, I drive a red Volkswagen Golf, and my username is dlee2021.")
        ],
        "expected": "VERIFIED"
    },
    
    # Email and Vehicle Details Verification
    {
        "client_details": {
            "full_name": "Lisa Martinez",
            "username": "lmartinez",
            "vehicle_registration": "MNO345GP",
            "vehicle_make": "Nissan",
            "vehicle_model": "Qashqai",
            "vehicle_color": "Grey",
            "email": "lisa.martinez@example.net",
            "passport_number": "8509195392089"
        },
        "messages": [
            AIMessage(content="Verify your account by providing specific details."),
            HumanMessage(content="I have a grey Nissan Qashqai, and my email is lisa.martinez@example.net.")
        ],
        "expected": "VERIFIED"
    },
    
    # Username and Vehicle Make Verification
    {
        "client_details": {
            "full_name": "Robert Anderson",
            "username": "randerson",
            "vehicle_registration": "PQR678WC",
            "vehicle_make": "Ford",
            "vehicle_model": "Ranger",
            "vehicle_color": "Blue",
            "email": "robert.a@example.org",
            "passport_number": "7703125392082"
        },
        "messages": [
            AIMessage(content="Provide account verification details."),
            HumanMessage(content="My username is randerson, and I drive a Ford Ranger.")
        ],
        "expected": "VERIFIED"
    },
    
    # Detailed ID Verification
    {
        "client_details": {
            "full_name": "Jennifer Williams",
            "username": "jwilliams",
            "vehicle_registration": "STU901EC",
            "vehicle_make": "Mercedes",
            "vehicle_model": "C-Class",
            "vehicle_color": "Silver",
            "email": "jennifer.w@example.com",
            "id_number": "8201015392084"
        },
        "messages": [
            AIMessage(content="Please confirm your ID number for verification."),
            HumanMessage(content="My ID number is 8201015392084.")
        ],
        "expected": "VERIFIED"
    },
    
    # Vehicle Registration and Email Verification
    {
        "client_details": {
            "full_name": "Thomas Wilson",
            "username": "twilson",
            "vehicle_registration": "VWX234MP",
            "vehicle_make": "Hyundai",
            "vehicle_model": "Tucson",
            "vehicle_color": "Green",
            "email": "thomas.wilson@example.net",
            "passport_number": "7409125392088"
        },
        "messages": [
            AIMessage(content="Verify your account with details."),
            HumanMessage(content="My registration is VWX234MP, and my email is thomas.wilson@example.net.")
        ],
        "expected": "INSUFFICIENT_INFO"
    },
    
    # Comprehensive Multi-Field Verification
    {
        "client_details": {
            "full_name": "Emily Davis",
            "username": "edavis",
            "vehicle_registration": "YZA567NW",
            "vehicle_make": "Mazda",
            "vehicle_model": "CX-5",
            "vehicle_color": "Brown",
            "email": "emily.davis@example.org",
            "passport_number": "7805015392080"
        },
        "messages": [
            AIMessage(content="Provide multiple account details for verification."),
            HumanMessage(content="My username is edavis, I drive a brown Mazda CX-5, and my registration is YZA567NW.")
        ],
        "expected": "VERIFIED"
    },
    
    # ========== INSUFFICIENT_INFO CASES (10 cases) ==========
    # These cases remain unchanged as they don't depend on failure pattern detection
    
    # Vague Vehicle Description
    {
        "client_details": {
            "full_name": "Mark Johnson",
            "username": "mjohnson",
            "vehicle_registration": "ABC987XY",
            "vehicle_make": "Honda",
            "vehicle_model": "Accord",
            "vehicle_color": "Blue",
            "email": "mark.j@example.com",
            "passport_number": "8605125392086"
        },
        "messages": [
            AIMessage(content="Please provide specific account details."),
            HumanMessage(content="I have a car.")
        ],
        "expected": "INSUFFICIENT_INFO"
    },
    
    # Partial Email Matching
    {
        "client_details": {
            "full_name": "Rachel Green",
            "username": "rgreen",
            "vehicle_registration": "DEF654WC",
            "vehicle_make": "Audi",
            "vehicle_model": "Q5",
            "vehicle_color": "Red",
            "email": "rachel.green@example.org",
            "passport_number": "7702155392083"
        },
        "messages": [
            AIMessage(content="Verify your account details."),
            HumanMessage(content="My email has 'rachel' in it.")
        ],
        "expected": "INSUFFICIENT_INFO"
    },
    
    # Incomplete Username
    {
        "client_details": {
            "full_name": "Alex Turner",
            "username": "aturner2023",
            "vehicle_registration": "GHI321MP",
            "vehicle_make": "Volkswagen",
            "vehicle_model": "Passat",
            "vehicle_color": "White",
            "email": "alex.turner@example.net",
            "passport_number": "8907235392085"
        },
        "messages": [
            AIMessage(content="Confirm your account details."),
            HumanMessage(content="My username has 'turner' in it.")
        ],
        "expected": "INSUFFICIENT_INFO"
    },
    
    # Ambiguous Vehicle Details
    {
        "client_details": {
            "full_name": "Sophie Miller",
            "username": "smiller",
            "vehicle_registration": "JKL654NW",
            "vehicle_make": "Toyota",
            "vehicle_model": "RAV4",
            "vehicle_color": "Silver",
            "email": "sophie.miller@example.com",
            "passport_number": "7509195392089"
        },
        "messages": [
            AIMessage(content="Provide specific verification details."),
            HumanMessage(content="I drive a Japanese SUV.")
        ],
        "expected": "INSUFFICIENT_INFO"
    },
    
    # Minimal Information
    {
        "client_details": {
            "full_name": "Chris Brown",
            "username": "cbrown",
            "vehicle_registration": "MNO987EC",
            "vehicle_make": "Ford",
            "vehicle_model": "Focus",
            "vehicle_color": "Black",
            "email": "chris.brown@example.org",
            "passport_number": "8203015392084"
        },
        "messages": [
            AIMessage(content="Verify your identity with account details."),
            HumanMessage(content="I'm not sure what details you need.")
        ],
        "expected": "INSUFFICIENT_INFO"
    },
    
    # Partial Vehicle Description
    {
        "client_details": {
            "full_name": "Anna Wilson",
            "username": "awilson",
            "vehicle_registration": "PQR321WC",
            "vehicle_make": "BMW",
            "vehicle_model": "3 Series",
            "vehicle_color": "Grey",
            "email": "anna.wilson@example.net",
            "passport_number": "7606125392082"
        },
        "messages": [
            AIMessage(content="Confirm your vehicle details."),
            HumanMessage(content="I have a grey car.")
        ],
        "expected": "INSUFFICIENT_INFO"
    },
    
    # Incorrect Information Attempt
    {
        "client_details": {
            "full_name": "Ryan Martinez",
            "username": "rmartinez",
            "vehicle_registration": "STU654MP",
            "vehicle_make": "Nissan",
            "vehicle_model": "Altima",
            "vehicle_color": "Blue",
            "email": "ryan.martinez@example.com",
            "passport_number": "8704235392087"
        },
        "messages": [
            AIMessage(content="Verify your account details."),
            HumanMessage(content="I'm not entirely sure about my details.")
        ],
        "expected": "INSUFFICIENT_INFO"
    },
    
    # Generic Response
    {
        "client_details": {
            "full_name": "Laura Thompson",
            "username": "lthompson",
            "vehicle_registration": "VWX987NW",
            "vehicle_make": "Hyundai",
            "vehicle_model": "Elantra",
            "vehicle_color": "White",
            "email": "laura.thompson@example.org",
            "passport_number": "7808015392080"
        },
        "messages": [
            AIMessage(content="Please provide specific account verification details."),
            HumanMessage(content="What kind of details do you need?")
        ],
        "expected": "INSUFFICIENT_INFO"
    },
    
    # Minimal Vehicle Information
    {
        "client_details": {
            "full_name": "Daniel Lee",
            "username": "dlee",
            "vehicle_registration": "YZA321EC",
            "vehicle_make": "Mercedes",
            "vehicle_model": "E-Class",
            "vehicle_color": "Black",
            "email": "daniel.lee@example.net",
            "passport_number": "8105125392086"
        },
        "messages": [
            AIMessage(content="Verify your identity with specific details."),
            HumanMessage(content="I drive a car.")
        ],
        "expected": "INSUFFICIENT_INFO"
    },
    
    # Incomplete Email Verification
    {
        "client_details": {
            "full_name": "Jessica Rodriguez",
            "username": "jrodriguez",
            "vehicle_registration": "BCD654WC",
            "vehicle_make": "Kia",
            "vehicle_model": "Optima",
            "vehicle_color": "Red",
            "email": "jessica.rodriguez@example.com",
            "id_number": "7702155392083"
        },
        "messages": [
            AIMessage(content="Confirm your email address."),
            HumanMessage(content="My email is from example.")
        ],
        "expected": "INSUFFICIENT_INFO"
    },
    
    # ========== VERIFICATION_FAILED CASES (10 cases) ==========
    # These cases need to be updated to focus on attempt count rather than behavior patterns
    
    # Multiple Verification Attempts (Exceeding max_failed_attempts)
    {
        "client_details": {
            "full_name": "Michael Taylor",
            "username": "mtaylor",
            "vehicle_registration": "EFG789NW",
            "vehicle_make": "Dodge",
            "vehicle_model": "Charger",
            "vehicle_color": "Red",
            "email": "michael.taylor@example.com",
            "id_number": "8305125392086"
        },
        "messages": [
            AIMessage(content="Please verify your account details."),
            HumanMessage(content="I have a red car."),
            AIMessage(content="Could you provide more specific information?"),
            HumanMessage(content="I drive a Toyota."),
            AIMessage(content="We're having trouble verifying your identity."),
            HumanMessage(content="I drive a car.")
        ],
        "expected": "VERIFICATION_FAILED",
        "verification_attempts": 3  # Explicitly setting the attempt count
    },
    
    # Multiple Verification Attempts
    {
        "client_details": {
            "full_name": "Sarah Williams",
            "username": "swilliams",
            "vehicle_registration": "HIJ456EC",
            "vehicle_make": "Mazda",
            "vehicle_model": "CX-3",
            "vehicle_color": "Silver",
            "email": "sarah.williams@example.net",
            "id_number": "7602155392083"
        },
        "messages": [
            AIMessage(content="Please verify your identity."),
            HumanMessage(content="Why do you need my details?"),
            AIMessage(content="We need to confirm your account information."),
            HumanMessage(content="I'm not sure about that."),
            AIMessage(content="This is a standard security procedure."),
            HumanMessage(content="I have a car.")
        ],
        "expected": "VERIFICATION_FAILED",
        "verification_attempts": 3
    },
    
    # Incorrect Information with Multiple Attempts
    {
        "client_details": {
            "full_name": "David Rodriguez",
            "username": "drodriguez",
            "vehicle_registration": "KLM321WC",
            "vehicle_make": "Ford",
            "vehicle_model": "Mustang",
            "vehicle_color": "Blue",
            "email": "david.rodriguez@example.org",
            "id_number": "8907235392085"
        },
        "messages": [
            AIMessage(content="Please confirm your account details."),
            HumanMessage(content="My ID is 1234567890."),
            AIMessage(content="That doesn't match our records. Could you verify again?"),
            HumanMessage(content="My registration is XYZ000."),
            AIMessage(content="These details are incorrect. Final verification attempt."),
            HumanMessage(content="My email is david@example.com")
        ],
        "expected": "VERIFICATION_FAILED",
        "verification_attempts": 4
    },
    
    # Multiple Verification Attempts
    {
        "client_details": {
            "full_name": "Jennifer Lee",
            "username": "jlee",
            "vehicle_registration": "NOP654MP",
            "vehicle_make": "Honda",
            "vehicle_model": "CR-V",
            "vehicle_color": "Green",
            "email": "jennifer.lee@example.com",
            "id_number": "7509195392089"
        },
        "messages": [
            AIMessage(content="We need to verify your identity."),
            HumanMessage(content="How do I know you're legitimate?"),
            AIMessage(content="We're following security protocols."),
            HumanMessage(content="What kind of information do you need?"),
            AIMessage(content="Please provide some verification details."),
            HumanMessage(content="I'm not sure I should.")
        ],
        "expected": "VERIFICATION_FAILED",
        "verification_attempts": 3
    },
    
    # Multiple Verification Attempts with Incomplete Information
    {
        "client_details": {
            "full_name": "Robert Chen",
            "username": "rchen",
            "vehicle_registration": "QRS987NW",
            "vehicle_make": "Toyota",
            "vehicle_model": "Camry",
            "vehicle_color": "White",
            "email": "robert.chen@example.net",
            "id_number": "8203015392084"
        },
        "messages": [
            AIMessage(content="Verify your account details."),
            HumanMessage(content="I drive a red car."),
            AIMessage(content="Could you confirm your registration?"),
            HumanMessage(content="I think it starts with A."),
            AIMessage(content="That doesn't match our records."),
            HumanMessage(content="I'm not sure what else to say.")
        ],
        "expected": "VERIFICATION_FAILED",
        "verification_attempts": 3
    },
    
    # Multiple Verification Attempts with Insufficient Responses
    {
        "client_details": {
            "full_name": "Emma Thompson",
            "username": "ethompson",
            "vehicle_registration": "TUV321EC",
            "vehicle_make": "Nissan",
            "vehicle_model": "Rogue",
            "vehicle_color": "Black",
            "email": "emma.thompson@example.org",
            "id_number": "7808015392080"
        },
        "messages": [
            AIMessage(content="We need to verify your identity."),
            HumanMessage(content="Okay."),
            AIMessage(content="Please provide some account details."),
            HumanMessage(content="What details?"),
            AIMessage(content="This is your final chance to verify."),
            HumanMessage(content="I'm not sure.")
        ],
        "expected": "VERIFICATION_FAILED",
        "verification_attempts": 3
    },
    
    # Multiple Failed Verification Attempts
    {
        "client_details": {
            "full_name": "Alex Morgan",
            "username": "amorgan",
            "vehicle_registration": "WXY654WC",
            "vehicle_make": "Hyundai",
            "vehicle_model": "Sonata",
            "vehicle_color": "Silver",
            "email": "alex.morgan@example.com",
            "id_number": "8104235392087"
        },
        "messages": [
            AIMessage(content="Verify your account details."),
            HumanMessage(content="I'm the account owner."),
            AIMessage(content="Can you provide specific details?"),
            HumanMessage(content="I have an account with you."),
            AIMessage(content="We need specific information to verify your identity."),
            HumanMessage(content="I should be in your system.")
        ],
        "expected": "VERIFICATION_FAILED",
        "verification_attempts": 3
    },
    
    # Multiple Attempts with Generic Responses
    {
        "client_details": {
            "full_name": "Sophie Carter",
            "username": "scarter",
            "vehicle_registration": "ZAB987NW",
            "vehicle_make": "Mercedes",
            "vehicle_model": "GLA",
            "vehicle_color": "Blue",
            "email": "sophie.carter@example.net",
            "id_number": "7606125392082"
        },
        "messages": [
            AIMessage(content="Please verify your identity."),
            HumanMessage(content="What kind of verification?"),
            AIMessage(content="We need specific account details."),
            HumanMessage(content="Like what?"),
            AIMessage(content="Can you confirm your vehicle or ID?"),
            HumanMessage(content="I'm not sure what details you need.")
        ],
        "expected": "VERIFICATION_FAILED",
        "verification_attempts": 3
    },
    
    # Multiple Verification Attempts
    {
        "client_details": {
            "full_name": "Ryan Peterson",
            "username": "rpeterson",
            "vehicle_registration": "CDE321MP",
            "vehicle_make": "Kia",
            "vehicle_model": "Sorento",
            "vehicle_color": "Red",
            "email": "ryan.peterson@example.org",
            "id_number": "8205125392086"
        },
        "messages": [
            AIMessage(content="We need to verify your account."),
            HumanMessage(content="Why do you need this information?"),
            AIMessage(content="This is a standard security procedure."),
            HumanMessage(content="I'm hesitant to share personal details."),
            AIMessage(content="Please provide verification details."),
            HumanMessage(content="I'm still not comfortable.")
        ],
        "expected": "VERIFICATION_FAILED",
        "verification_attempts": 3
    },
    
    # Multiple Verification Attempts with No Progress
    {
        "client_details": {
            "full_name": "Laura Martinez",
            "username": "lmartinez",
            "vehicle_registration": "FGH654WC",
            "vehicle_make": "BMW",
            "vehicle_model": "X3",
            "vehicle_color": "White",
            "email": "laura.martinez@example.com",
            "id_number": "7702155392083"
        },
        "messages": [
            AIMessage(content="We need to verify your identity."),
            HumanMessage(content="I'm not sure."),
            AIMessage(content="Please provide some account details."),
            HumanMessage(content="What details do you need?"),
            AIMessage(content="This is your final verification attempt."),
            HumanMessage(content="I don't know what to say.")
        ],
        "expected": "VERIFICATION_FAILED",
        "verification_attempts": 4  # Explicitly set to exceed the default max_failed_attempts
    }
]

