from langchain_core.messages import HumanMessage, AIMessage
from typing import Any
import time
import json


# Color formatting for terminal output
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

class ClientNameVerificationTester:
    """A test harness for evaluating the client name verification tool against expected classifications."""
    
    def __init__(self, verify_client_name: Any, default_max_failed_attempts=3, verbose=False):
        """
        Initialize the client name verification tester with specified settings.
        
        Args:
            verify_client_name: The verification function or tool to be tested
            default_max_failed_attempts (int, optional): Default maximum allowed failed 
                verification attempts before a test is considered failed. Defaults to 3.
            verbose (bool, optional): Whether to display detailed output during test 
                execution. Defaults to False.
        
        Attributes:
            test_results (list): Stores results of all executed tests
        """
        self.verify_client_name = verify_client_name
        self.default_max_failed_attempts = default_max_failed_attempts
        self.test_results = []
        self.verbose = verbose
        
    def run_test(self, client_full_name, messages, expected_result=None, 
                max_failed_attempts=None, test_number=None):
        """
        Run a single test case for the client verification tool.
        
        Args:
            client_full_name: The client name to verify against
            messages: List of message objects (HumanMessage or AIMessage)
            expected_result: Expected classification (optional)
            max_failed_attempts: Custom max failed attempts for this test (optional)
            test_number: Test case number for display
            
        Returns:
            The verification result object
        """
        test_max_failed_attempts = max_failed_attempts if max_failed_attempts is not None else self.default_max_failed_attempts
        
        # Record test case details
        test_case = {
            "client_name": client_full_name,
            "messages": [{"role": "assistant" if isinstance(m, AIMessage) else "human", 
                         "content": m.content} for m in messages],
            "max_failed_attempts": test_max_failed_attempts
        }
        
        # Start timer
        start_time = time.time()
        
        try:
            # Run verification
            invoke_params = {
                'client_full_name': client_full_name,
                'messages': messages,
                'max_failed_attempts': test_max_failed_attempts
            }
            
            # Call verify_client_name directly with the parameters it expects
            result = self.verify_client_name.invoke(invoke_params)
                
        except Exception as e:
            print(f"{COLORS['ERROR']}Error invoking verify_client_name: {e}{COLORS['END']}")
            result = {
                "classification": "ERROR",
                "name_variants_detected": [],
                "verification_attempts": 0,
                "error": str(e)
            }
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Check if result matches expected result (if provided)
        passed = (expected_result is None or result['classification'] == expected_result)
        
        # Store test result with adapters for missing fields (for backwards compatibility)
        test_result = {
            "test_case": test_case,
            "result": {
                "classification": result['classification'],
                "reasoning": result.get('error', "Reasoning not provided in optimized version"),
                "name_variants_detected": result.get('name_variants_detected', []),
                "verification_attempts": result.get('verification_attempts', 0),
                "failed_attempts": 0,  # Not tracked in optimized version
                "suggested_response": "Not provided in optimized version"  # Not included in optimized version
            },
            "metrics": {},  # Not tracked in optimized version
            "execution_time": execution_time,
            "passed": passed,
            "expected": expected_result,
            "test_number": test_number
        }
        self.test_results.append(test_result)
        
        # Print test result
        self._print_test_result_optimized(test_result, test_number)
        
        return result
    
    def run_all_tests(self, test_cases):
        """
        Run all predefined test cases with optimized output.
        
        Args:
            test_cases (list): List of test case dictionaries
        """
        print(f"\n{COLORS['HEADER']}{COLORS['BOLD']}==== RUNNING {len(test_cases)} VERIFICATION TESTS ===={COLORS['END']}\n")
        
        # Run all test cases
        for i, case in enumerate(test_cases):
            # Convert message objects if needed
            messages = case["messages"]
            if not isinstance(messages[0], (HumanMessage, AIMessage)):
                # Convert dictionary format to message objects
                converted_messages = []
                for msg in messages:
                    if msg["role"] == "assistant":
                        converted_messages.append(AIMessage(content=msg["content"]))
                    else:
                        converted_messages.append(HumanMessage(content=msg["content"]))
                messages = converted_messages
            
            result = self.run_test(
                client_full_name=case["name"],
                messages=messages,
                expected_result=case["expected"],
                max_failed_attempts=case.get("max_failed_attempts"),
                test_number=i+1,
            )
        
        # Print optimized summary
        self._print_summary_optimized()

    def _print_test_result_optimized(self, test_result, test_number=None):
        """
        Print a simplified, formatted result for a single test case with proper box alignment.
        
        Args:
            test_result (dict): The test result to print
            test_number (int, optional): Number of the test case
        """
        # Set a consistent width for all boxes
        width = 70
        
        # Determine test result status for the header
        passed = test_result['passed']
        status_text = "PASS ✓" if passed else "FAIL ✗"
        
        # Create a horizontal border line
        border_line = "─" * width
        
        # Print separation line between test cases
        print(f"\n{COLORS['HEADER']}{'═' * (width + 2)}{COLORS['END']}")
        
        # ===== HEADER BOX =====
        test_num_prefix = f"[{test_number}] " if test_number is not None else ""
        header_text = f"TEST CASE: {test_num_prefix}{test_result['test_case']['client_name']}"
        status_color = COLORS['SUCCESS'] if passed else COLORS['ERROR']
        
        # Print the header box with pink background
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
            "THIRD_PARTY": COLORS['WARNING'],
            "WRONG_PERSON": COLORS['ERROR'],
            "INSUFFICIENT_INFO": COLORS['WARNING'],
            "UNAVAILABLE": COLORS['WARNING'],
            "VERIFICATION_FAILED": COLORS['ERROR'],
            "ERROR": COLORS['ERROR']
        }.get(test_result['result']['classification'], COLORS['INFO'])
        
        expected_color = {
            "VERIFIED": COLORS['SUCCESS'],
            "THIRD_PARTY": COLORS['WARNING'],
            "WRONG_PERSON": COLORS['ERROR'],
            "INSUFFICIENT_INFO": COLORS['WARNING'],
            "UNAVAILABLE": COLORS['WARNING'],
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
        
        # Third line with verification attempts count
        verification_attempts = test_result['result']['verification_attempts']
        va_section = f"Verification Attempts: {verification_attempts}"
        
        print(f"│ {va_section}{' ' * (width - len(va_section) - 2)}│")
        print(f"└{border_line}┘")
        
        # ----- NAME VARIANTS SECTION -----
        print(f"\n{COLORS['BOLD']}DETECTED NAME VARIANTS:{COLORS['END']}")
        name_variants = test_result['result']['name_variants_detected']
        
        if name_variants:
            for variant in name_variants:
                print(f"  {COLORS['INFO']}✓{COLORS['END']} {variant}")
        else:
            print(f"  {COLORS['WARNING']}No name variants detected{COLORS['END']}")
            
        # Print execution time as a footer
        print(f"\n{COLORS['INFO']}Execution time: {test_result['execution_time']:.3f}s{COLORS['END']}")
        
    def _print_summary_optimized(self):
        """Print a more concise summary of all test results."""
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
        
        # Classification distribution
        print(f"\n{COLORS['BOLD']}Classification Distribution:{COLORS['END']}")
        
        # Color mapping
        result_colors = {
            "VERIFIED": COLORS['SUCCESS'],
            "THIRD_PARTY": COLORS['WARNING'],
            "WRONG_PERSON": COLORS['ERROR'],
            "INSUFFICIENT_INFO": COLORS['WARNING'],
            "UNAVAILABLE": COLORS['WARNING'],
            "VERIFICATION_FAILED": COLORS['ERROR'],
            "ERROR": COLORS['ERROR']
        }
        
        # Increase first column width by 1.2x
        first_col_width = int(18 * 1.2)
        
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
        
        # Get unique classification values for matrix - focus on main classifications
        classifications = ["VERIFIED", "THIRD_PARTY", "WRONG_PERSON", "INSUFFICIENT_INFO", "UNAVAILABLE", "VERIFICATION_FAILED"]
        
        # Print header with increased first column width
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

        # Add these lines to print failed test indices
        failed_indices = [r["test_number"] for r in self.test_results if not r["passed"]]
        if failed_indices:
            print(f"\n{COLORS['ERROR']}Failed Test Indices: {', '.join(map(str, failed_indices))}{COLORS['END']}")
        
        # Group failures by expected vs actual classification
        failure_types = {}
        for r in self.test_results:
            if not r["passed"]:
                key = f"{r['expected']} → {r['result']['classification']}"
                if key not in failure_types:
                    failure_types[key] = []
                failure_types[key].append(r["test_number"])
        
        if failure_types:
            print(f"\n{COLORS['ERROR']}Failures by Classification Type:{COLORS['END']}")
            for key, indices in failure_types.items():
                print(f"  {key}: Tests {', '.join(map(str, indices))}")

    def export_results(self, filename="client_name_verification_test_results.json"):
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
                f.write("# Client Name Verification Test Report\n\n")
                
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
                    f.write(f"### Test Case {i}: {result['test_case']['client_name']}\n\n")
                    f.write(f"- **Classification:** `{result['result']['classification']}`\n")
                    f.write(f"- **Expected Result:** `{result['expected']}`\n")
                    f.write(f"- **Status:** {'✅ PASSED' if result['passed'] else '❌ FAILED'}\n")
                    f.write(f"- **Verification Attempts:** {result['result']['verification_attempts']}\n\n")
                    
                    # Name variants
                    name_variants = result['result']['name_variants_detected']
                    f.write("#### Detected Name Variants\n")
                    if name_variants:
                        f.write("- " + "\n- ".join(name_variants) + "\n\n")
                    else:
                        f.write("- *No name variants detected*\n\n")
                
                # Classification Distribution
                f.write("## Classification Distribution\n")
                classification_counts = {}
                for r in self.test_results:
                    cls = r['result']['classification']
                    classification_counts[cls] = classification_counts.get(cls, 0) + 1
                
                for cls, count in classification_counts.items():
                    f.write(f"- **{cls}:** {count} ({count/total_tests*100:.2f}%)\n")
                
                # Add failure breakdown by type
                failed_indices = [r["test_number"] for r in self.test_results if not r["passed"]]
                if failed_indices:
                    f.write("\n## Failed Test Cases\n")
                    f.write(f"- Failed Test Indices: {', '.join(map(str, failed_indices))}\n\n")
                    
                    # Group failures by expected vs actual classification
                    failure_types = {}
                    for r in self.test_results:
                        if not r["passed"]:
                            key = f"{r['expected']} → {r['result']['classification']}"
                            if key not in failure_types:
                                failure_types[key] = []
                            failure_types[key].append(r["test_number"])
                    
                    f.write("### Failures by Classification Type\n")
                    for key, indices in failure_types.items():
                        f.write(f"- **{key}**: Tests {', '.join(map(str, indices))}\n")
            
            print(f"\n{COLORS['SUCCESS']}Simplified report generated at {COLORS['BOLD']}{filename}{COLORS['END']}")
        except Exception as e:
            print(f"\n{COLORS['ERROR']}Error generating report: {e}{COLORS['END']}")

#########################################################################################
# Test cases
#########################################################################################

test_cases = [
    # ========== VERIFIED CASES ==========
    # Cases where the person's identity is successfully verified
    {
        "name": "John Smith",
        "messages": [
            AIMessage(content="Hello, am I speaking with John Smith?"),
            HumanMessage(content="Yes, this is John Smith speaking.")
        ],
        "expected": "VERIFIED"
    },
    {
        "name": "Thomas Anderson",
        "messages": [
            AIMessage(content="Hello, am I speaking with Thomas Anderson?"),
            HumanMessage(content="Yeah, this is Tom. What's this regarding?")
        ],
        "expected": "VERIFIED"
    },
    {
        "name": "Jennifer Lopez",
        "messages": [
            AIMessage(content="Hello, is this Jennifer Lopez?"),
            HumanMessage(content="Umm, yes... that's me. Who's calling please?")
        ],
        "expected": "VERIFIED"
    },
    {
        "name": "Robert William Johnson",
        "messages": [
            AIMessage(content="Hello, am I speaking with Robert William Johnson?"),
            HumanMessage(content="This is Rob Johnson, yes.")
        ],
        "expected": "VERIFIED"
    },
    {
        "name": "Alexander Wang",
        "messages": [
            AIMessage(content="Hello, am I speaking with Alexander Wang?"),
            HumanMessage(content="Yes, it's Alex speaking.")
        ],
        "expected": "VERIFIED"
    },
    {
        "name": "Elizabeth Rodriguez",
        "messages": [
            AIMessage(content="Hello, am I speaking with Elizabeth Rodriguez?"),
            HumanMessage(content="This is Liz Rodriguez speaking.")
        ],
        "expected": "VERIFIED"
    },
    {
        "name": "William Montgomery",
        "messages": [
            AIMessage(content="Hello, am I speaking with William Montgomery?"),
            HumanMessage(content="Yes, you've reached Bill. How can I help?")
        ],
        "expected": "VERIFIED"
    },
    {
        "name": "Samantha Chen",
        "messages": [
            AIMessage(content="Hello, am I speaking with Samantha Chen?"),
            HumanMessage(content="Speaking. I've been expecting your call about my account.")
        ],
        "expected": "VERIFIED"
    },
    {
        "name": "Christopher O'Neill",
        "messages": [
            AIMessage(content="Hello, am I speaking with Christopher O'Neill?"),
            HumanMessage(content="Yes, that's me, but I go by Chris.")
        ],
        "expected": "VERIFIED"
    },
    
    # ========== THIRD_PARTY CASES ==========
    # Cases where someone else is calling on behalf of the named person
    {
        "name": "Maria Garcia",
        "messages": [
            AIMessage(content="Hello, am I speaking with Maria Garcia?"),
            HumanMessage(content="No, I'm her husband calling on her behalf.")
        ],
        "expected": "THIRD_PARTY"
    },
    {
        "name": "David Wilson",
        "messages": [
            AIMessage(content="Hello, am I speaking with David Wilson?"),
            HumanMessage(content="This is his attorney, James Brown. I'm calling regarding his account.")
        ],
        "expected": "THIRD_PARTY"
    },
    {
        "name": "Nguyen Thanh Trung",
        "messages": [
            AIMessage(content="Hello, may I speak to Nguyen Thanh Trung?"),
            HumanMessage(content="No, I am calling on behalf of him.")
        ],
        "expected": "THIRD_PARTY"
    },
    {
        "name": "Samuel Jackson",
        "messages": [
            AIMessage(content="Hello, am I speaking with Samuel Jackson?"),
            HumanMessage(content="No, this is his assistant, Alice. I handle Mr. Jackson's accounts.")
        ],
        "expected": "THIRD_PARTY"
    },
    {
        "name": "Olivia Wilson",
        "messages": [
            AIMessage(content="Hello, am I speaking with Olivia Wilson?"),
            HumanMessage(content="I'm her daughter. Mom asked me to call about her recent statement.")
        ],
        "expected": "THIRD_PARTY"
    },
    
    # ========== WRONG_PERSON CASES ==========
    # Cases where the person reached is definitely not the intended recipient
    {
        "name": "Robert Johnson",
        "messages": [
            AIMessage(content="Hello, am I speaking with Robert Johnson?"),
            HumanMessage(content="No, you have the wrong number. I don't know anyone by that name.")
        ],
        "expected": "WRONG_PERSON"
    },
    {
        "name": "Amanda Williams",
        "messages": [
            AIMessage(content="Hello, am I speaking with Amanda Williams?"),
            HumanMessage(content="No, there's no Amanda here. You must have dialed incorrectly.")
        ],
        "expected": "WRONG_PERSON"
    },
    {
        "name": "Jessica Parker",
        "messages": [
            AIMessage(content="Hello, am I speaking with Jessica Parker?"),
            HumanMessage(content="No, this is Jennifer Parker. Jessica is my sister but she doesn't live here.")
        ],
        "expected": "THIRD_PARTY"
    },
    {
        "name": "George Smith Jr.",
        "messages": [
            AIMessage(content="Hello, am I speaking with George Smith Junior?"),
            HumanMessage(content="No, this is George Smith Senior. Junior is my son and he has his own phone number.")
        ],
        "expected": "THIRD_PARTY"
    },
    {
        "name": "Daniel Martinez",
        "messages": [
            AIMessage(content="Hello, am I speaking with Daniel Martinez?"),
            HumanMessage(content="You've got the wrong person. My name is David Martinez, no relation.")
        ],
        "expected": "WRONG_PERSON"
    },
    
    # ========== INSUFFICIENT_INFO CASES ==========
    # Cases where identity cannot be determined from the response
    {
        "name": "Sarah Williams",
        "messages": [
            AIMessage(content="Hello, am I speaking with Sarah Williams?"),
            HumanMessage(content="I'm calling about the account.")
        ],
        "expected": "INSUFFICIENT_INFO"
    },
    {
        "name": "Michelle Johnson",
        "messages": [
            AIMessage(content="Hello, am I speaking with Michelle Johnson?"),
            HumanMessage(content="Who's calling?"),
            AIMessage(content="This is ABC Bank regarding your account."),
            HumanMessage(content="Oh, okay. What do you need?")
        ],
        "expected": "INSUFFICIENT_INFO"
    },
    {
        "name": "Brandon Taylor",
        "messages": [
            AIMessage(content="Hello, am I speaking with Brandon Taylor?"),
            HumanMessage(content="What is this regarding?")
        ],
        "expected": "INSUFFICIENT_INFO"
    },
    {
        "name": "Emma Richardson",
        "messages": [
            AIMessage(content="Hello, am I speaking with Emma Richardson?"),
            HumanMessage(content="Why do you need to know? What company are you with?")
        ],
        "expected": "INSUFFICIENT_INFO"
    },
    
    # ========== UNAVAILABLE CASES ==========
    # Cases where the person confirms their identity but cannot talk right now
    {
        "name": "Benjamin Wright",
        "messages": [
            AIMessage(content="Hello, am I speaking with Benjamin Wright?"),
            HumanMessage(content="Yes, this is Ben. I'm in the middle of something. Can you call back in an hour?")
        ],
        "expected": "UNAVAILABLE"
    },
    {
        "name": "Andrew Peterson",
        "messages": [
            AIMessage(content="Hello, am I speaking with Andrew Peterson?"),
            HumanMessage(content="Yes, this is Andrew, but I'm in a meeting right now. Can you call back in an hour?")
        ],
        "expected": "UNAVAILABLE"
    },
    {
        "name": "Rebecca Martinez",
        "messages": [
            AIMessage(content="Hello, is this Rebecca Martinez?"),
            HumanMessage(content="Yes, that's me, but I'm driving at the moment. Can we talk later?")
        ],
        "expected": "UNAVAILABLE"
    },
    {
        "name": "Jonathan Kim",
        "messages": [
            AIMessage(content="Hello, am I speaking with Jonathan Kim?"),
            HumanMessage(content="Yeah, it's Jon. I'm really busy right now though. Please call tomorrow.")
        ],
        "expected": "UNAVAILABLE"
    },
    
    # ========== VERIFICATION_FAILED CASES ==========
    # Cases where verification should fail after multiple attempts
    # Original verification failed cases with max_failed_attempts parameter
    {
        "name": "Michael Brown",
        "messages": [
            AIMessage(content="Hello, am I speaking with Michael Brown?"),
            HumanMessage(content="Who wants to know?"),
            AIMessage(content="This is ABC Bank. We need to verify we're speaking with Michael Brown."),
            HumanMessage(content="I don't give out personal information on the phone."),
            AIMessage(content="I understand your concern. To proceed with account matters, I need to confirm your identity."),
            HumanMessage(content="Just tell me what this is about first.")
        ],
        "expected": "VERIFICATION_FAILED",
        "max_failed_attempts": 3
    },
    {
        "name": "Jennifer Lee",
        "messages": [
            AIMessage(content="Hello, am I speaking with Jennifer Lee?"),
            HumanMessage(content="Maybe. What's this about?"),
            AIMessage(content="This is regarding your recent application. I need to confirm I'm speaking with Jennifer Lee."),
            HumanMessage(content="I didn't apply for anything."),
            AIMessage(content="I apologize for any confusion. Could you confirm if you are Jennifer Lee?"),
            HumanMessage(content="Look, I'm not going to confirm anything until you tell me exactly what this is about.")
        ],
        "expected": "VERIFICATION_FAILED",
        "max_failed_attempts": 3
    },
    {
        "name": "David Rodriguez",
        "messages": [
            AIMessage(content="Hello, am I speaking with David Rodriguez?"),
            HumanMessage(content="Hmm, who's asking?"),
            AIMessage(content="This is ABC Bank calling about your account. Can you confirm you're David Rodriguez?"),
            HumanMessage(content="I don't have an account with ABC."),
            AIMessage(content="I apologize if there's confusion. To clarify, can you confirm if you are David Rodriguez?"),
            HumanMessage(content="I told you I don't have an account with you. I think you have the wrong person.")
        ],
        "expected": "VERIFICATION_FAILED",
        "max_failed_attempts": 3
    },
    
    
    # ========== ADDITIONAL VERIFICATION_FAILED CASES ==========
    # New test cases for VERIFICATION_FAILED scenarios
    
    # Case 1: Suspicious responses pattern - simulated multiple failed attempts
    {
        "name": "Richard Thompson",
        "messages": [
            AIMessage(content="Hello, am I speaking with Richard Thompson?"),
            HumanMessage(content="Why do you need to know that?"),
            AIMessage(content="This is Secure Bank calling regarding some unusual activity. First, I need to confirm it's Richard Thompson."),
            HumanMessage(content="What unusual activity? I want details before I tell you anything."),
            AIMessage(content="For your protection, I need to verify your identity before discussing account details."),
            HumanMessage(content="I don't trust this call. You could be anyone.")
        ],
        "expected": "VERIFICATION_FAILED",
        "max_failed_attempts": 3
    },
    
    # Case 2: Explicit refusal to provide identity verification
    {
        "name": "Katherine Wilson",
        "messages": [
            AIMessage(content="Hello, am I speaking with Katherine Wilson?"),
            HumanMessage(content="I don't answer personal questions on unsolicited calls."),
            AIMessage(content="I understand your concern. This is Capital Finance. We're calling about your recent loan application."),
            HumanMessage(content="I'm not confirming or denying my identity to an unknown caller."),
            AIMessage(content="For security purposes, I need to verify I'm speaking with the right person."),
            HumanMessage(content="Sorry, but I'm not comfortable with this. Goodbye.")
        ],
        "expected": "VERIFICATION_FAILED",
        "max_failed_attempts": 3
    },
    
    # Case 3: Changing the subject and direct refusal pattern
    {
        "name": "Anthony Garcia",
        "messages": [
            AIMessage(content="Hello, am I speaking with Anthony Garcia?"),
            HumanMessage(content="Where did you get this number from?"),
            AIMessage(content="This is regarding your insurance policy. Can you confirm you're Anthony Garcia?"),
            HumanMessage(content="I don't recall having an insurance policy with your company."),
            AIMessage(content="Before I can discuss any details, can you confirm your identity?"),
            HumanMessage(content="I'd rather you just tell me what this is about first.")
        ],
        "expected": "VERIFICATION_FAILED",
        "max_failed_attempts": 3
    },
    
    # Case 4: Language barrier causing verification failure
    {
        "name": "Ming Zhang",
        "messages": [
            AIMessage(content="Hello, am I speaking with Ming Zhang?"),
            HumanMessage(content="Sorry, I not understand well. Who calling?"),
            AIMessage(content="This is ABC Bank. I need to confirm I'm speaking with Ming Zhang."),
            HumanMessage(content="Sorry, my English not good. What you want?"),
            AIMessage(content="Are you Ming Zhang? I need to verify your identity."),
            HumanMessage(content="I not understand. Call back when translator please.")
        ],
        "expected": "VERIFICATION_FAILED",
        "max_failed_attempts": 3
    },
    
    
    
    # Case 6: Verification failed with conflicting information
    {
        "name": "Eduardo Fernandez",
        "messages": [
            AIMessage(content="Hello, am I speaking with Eduardo Fernandez?"),
            HumanMessage(content="Who's asking?"),
            AIMessage(content="This is First National Bank regarding your mortgage application."),
            HumanMessage(content="I don't have a mortgage with First National."),
            AIMessage(content="I need to verify if you are Eduardo Fernandez before proceeding."),
            HumanMessage(content="Look, I've already told you I don't have an account with you. Why would I be Eduardo?")
        ],
        "expected": "VERIFICATION_FAILED",
        "max_failed_attempts": 3
    },
    
    # Case 7: Caller getting increasingly agitated
    {
        "name": "Sophia Martinez",
        "messages": [
            AIMessage(content="Hello, am I speaking with Sophia Martinez?"),
            HumanMessage(content="Who wants to know?"),
            AIMessage(content="This is ABC Credit Union regarding your account."),
            HumanMessage(content="How do I know you're really from my credit union?"),
            AIMessage(content="I understand your concern. For security, I need to verify I'm speaking with Sophia Martinez."),
            HumanMessage(content="No, YOU need to verify who YOU are first! This is ridiculous. Stop calling me!")
        ],
        "expected": "VERIFICATION_FAILED",
        "max_failed_attempts": 3
    },
    
    # Case 8: Silent treatment combined with evasion 
    {
        "name": "Jacob Williams",
        "messages": [
            AIMessage(content="Hello, am I speaking with Jacob Williams?"),
            HumanMessage(content="..."),  # Silence
            AIMessage(content="Hello? I'm trying to reach Jacob Williams. Is this Jacob?"),
            HumanMessage(content="I can hear you."),
            AIMessage(content="Great. Could you please confirm if you are Jacob Williams?"),
            HumanMessage(content="I'm listening. What is this regarding?")
        ],
        "expected": "VERIFICATION_FAILED",
        "max_failed_attempts": 3
    }
]

# test_cases = [test_cases[i] for i in [16, 22]]
# test_cases = test_cases[0:10]

# test_cases = [
#     # ========== VERIFIED CASES ==========
#     # Cases where the person's identity is successfully verified
#     {
#         "name": "John Smith",
#         "messages": [
#             AIMessage(content="Hello, am I speaking with John Smith?"),
#             HumanMessage(content="Yes, this is John Smith speaking.")
#         ],
#         "expected": "VERIFIED"
#     },
#     {
#         "name": "Thomas Anderson",
#         "messages": [
#             AIMessage(content="Hello, am I speaking with Thomas Anderson?"),
#             HumanMessage(content="Yeah, this is Tom. What's this regarding?")
#         ],
#         "expected": "VERIFIED"
#     },
#     {
#         "name": "Jennifer Lopez",
#         "messages": [
#             AIMessage(content="Hello, is this Jennifer Lopez?"),
#             HumanMessage(content="Umm, yes... that's me. Who's calling please?")
#         ],
#         "expected": "VERIFIED"
#     },
#     {
#         "name": "Robert Johnson",
#         "messages": [
#             AIMessage(content="May I speak with Robert Johnson?"),
#             HumanMessage(content="Speaking. What can I do for you?")
#         ],
#         "expected": "VERIFIED"
#     },
#     {
#         "name": "Sarah Williams",
#         "messages": [
#             AIMessage(content="Am I speaking with Sarah Williams?"),
#             HumanMessage(content="Yes, that's me. Who's calling?")
#         ],
#         "expected": "VERIFIED"
#     },
#     {
#         "name": "Michael Brown",
#         "messages": [
#             AIMessage(content="Is this Michael Brown?"),
#             HumanMessage(content="Mike here. How can I help you?")
#         ],
#         "expected": "VERIFIED"
#     },
    
#     # ========== THIRD_PARTY CASES ==========
#     # Cases where a third party is answering on behalf of the client
#     {
#         "name": "Patricia Garcia",
#         "messages": [
#             AIMessage(content="Hello, is Patricia Garcia available?"),
#             HumanMessage(content="This is her husband. She's not available right now.")
#         ],
#         "expected": "THIRD_PARTY"
#     },
#     {
#         "name": "David Wilson",
#         "messages": [
#             AIMessage(content="May I speak with David Wilson please?"),
#             HumanMessage(content="I'm his assistant. He's in a meeting. Can I take a message?")
#         ],
#         "expected": "THIRD_PARTY"
#     },
#     {
#         "name": "Elizabeth Taylor",
#         "messages": [
#             AIMessage(content="Is Elizabeth Taylor available?"),
#             HumanMessage(content="This is her daughter speaking. Mom is resting right now.")
#         ],
#         "expected": "THIRD_PARTY"
#     },
#     {
#         "name": "George Smith",
#         "messages": [
#             AIMessage(content="Hello, am I speaking with George Smith?"),
#             HumanMessage(content="No, this is George Smith Senior. Junior is my son.")
#         ],
#         "expected": "THIRD_PARTY"
#     },
#     {
#         "name": "Katherine Parker",
#         "messages": [
#             AIMessage(content="Is this Katherine Parker?"),
#             HumanMessage(content="No, this is her sister Karen. Katherine asked me to answer her calls today.")
#         ],
#         "expected": "THIRD_PARTY"
#     },
    
#     # ========== UNAVAILABLE CASES ==========
#     # Cases where the client confirms identity but cannot talk now
#     {
#         "name": "William Davis",
#         "messages": [
#             AIMessage(content="Hello, is this William Davis?"),
#             HumanMessage(content="Yes, that's me, but I'm driving right now. Can you call back later?")
#         ],
#         "expected": "UNAVAILABLE"
#     },
#     {
#         "name": "Maria Rodriguez",
#         "messages": [
#             AIMessage(content="Am I speaking with Maria Rodriguez?"),
#             HumanMessage(content="Speaking, but I'm in a meeting. Not a good time.")
#         ],
#         "expected": "UNAVAILABLE"
#     },
#     {
#         "name": "Christopher Lee",
#         "messages": [
#             AIMessage(content="Is this Christopher Lee?"),
#             HumanMessage(content="Yes it is, but I'm at work. Can this wait till this evening?")
#         ],
#         "expected": "UNAVAILABLE"
#     },
#     {
#         "name": "Amanda White",
#         "messages": [
#             AIMessage(content="Hello, Amanda White?"),
#             HumanMessage(content="This is Amanda, but I'm about to board a plane. Call me tomorrow.")
#         ],
#         "expected": "UNAVAILABLE"
#     },
#     {
#         "name": "Daniel Martinez",
#         "messages": [
#             AIMessage(content="Hello, Daniel Martinez?"),
#             HumanMessage(content="Yes that's me, but I'm busy at the moment with my kids.")
#         ],
#         "expected": "UNAVAILABLE"
#     },
    
#     # ========== WRONG_PERSON CASES ==========
#     # Cases where the wrong person is reached
#     {
#         "name": "Jessica Thompson",
#         "messages": [
#             AIMessage(content="Is this Jessica Thompson?"),
#             HumanMessage(content="No, you have the wrong number. I don't know anyone by that name.")
#         ],
#         "expected": "WRONG_PERSON"
#     },
#     {
#         "name": "Andrew Clark",
#         "messages": [
#             AIMessage(content="Hello, Andrew Clark?"),
#             HumanMessage(content="No, this is Simon Evans. There's no Andrew here.")
#         ],
#         "expected": "WRONG_PERSON"
#     },
#     {
#         "name": "Michelle Scott",
#         "messages": [
#             AIMessage(content="May I speak with Michelle Scott?"),
#             HumanMessage(content="Wrong number. Nobody by that name lives here.")
#         ],
#         "expected": "WRONG_PERSON"
#     },
#     {
#         "name": "Kevin Robinson",
#         "messages": [
#             AIMessage(content="Is Kevin Robinson available?"),
#             HumanMessage(content="You've got the wrong person, sorry.")
#         ],
#         "expected": "WRONG_PERSON"
#     },
    
#     # ========== INSUFFICIENT_INFO CASES ==========
#     # Cases where not enough information is provided
#     {
#         "name": "Laura Miller",
#         "messages": [
#             AIMessage(content="Is this Laura Miller?"),
#             HumanMessage(content="Who's calling please?")
#         ],
#         "expected": "INSUFFICIENT_INFO"
#     },
#     {
#         "name": "Brandon Adams",
#         "messages": [
#             AIMessage(content="Am I speaking with Brandon Adams?"),
#             HumanMessage(content="What is this regarding?")
#         ],
#         "expected": "INSUFFICIENT_INFO"
#     },
#     {
#         "name": "Samantha Hill",
#         "messages": [
#             AIMessage(content="Hello, is this Samantha Hill?"),
#             HumanMessage(content="I'm not confirming anything until I know who's calling.")
#         ],
#         "expected": "INSUFFICIENT_INFO"
#     },
#     {
#         "name": "Ryan Collins",
#         "messages": [
#             AIMessage(content="Is this Ryan Collins?"),
#             HumanMessage(content="Where did you get this number from?")
#         ],
#         "expected": "INSUFFICIENT_INFO"
#     },
    
#     # ========== VERIFICATION_FAILED CASES ==========
#     # Cases where multiple attempts at verification fail
#     {
#         "name": "Olivia Carter",
#         "messages": [
#             AIMessage(content="Hello, am I speaking with Olivia Carter?"),
#             HumanMessage(content="Who's asking?"),
#             AIMessage(content="This is Bank of America calling. I need to confirm I'm speaking with Olivia Carter."),
#             HumanMessage(content="What's this about?"),
#             AIMessage(content="It's regarding your account. For security purposes, could you confirm you're Olivia Carter?"),
#             HumanMessage(content="I'm not giving any information over the phone"),
#             AIMessage(content="I understand your concern, but I need to verify I'm speaking with the right person. Are you Olivia Carter?")
#         ],
#         "expected": "VERIFICATION_FAILED"
#     },
#     {
#         "name": "Ethan Morris",
#         "messages": [
#             AIMessage(content="Hello, is this Ethan Morris?"),
#             HumanMessage(content="Uh, who's calling?"),
#             AIMessage(content="This is ABC Financial. I need to speak with Ethan Morris, please."),
#             HumanMessage(content="What do you want?"),
#             AIMessage(content="I'm calling about an important matter, but first I need to confirm I'm speaking with Ethan Morris."),
#             HumanMessage(content="Not saying anything till you tell me what this is about"),
#             AIMessage(content="For privacy reasons, I need to verify your identity first. Can you confirm you're Ethan Morris?")
#         ],
#         "expected": "VERIFICATION_FAILED"
#     },
#     {
#         "name": "Sophia Martinez",
#         "messages": [
#             AIMessage(content="May I speak with Sophia Martinez?"),
#             HumanMessage(content="Mmm..."),
#             AIMessage(content="Hello? Is this Sophia Martinez I'm speaking with?"),
#             HumanMessage(content="Maybe, maybe not."),
#             AIMessage(content="I understand your hesitation, but I do need to confirm I'm speaking with Sophia Martinez."),
#             HumanMessage(content="And if it is?"),
#             AIMessage(content="I'm calling about a private matter, but I first need to verify I'm speaking with Sophia Martinez.")
#         ],
#         "expected": "VERIFICATION_FAILED"
#     }
# ]


