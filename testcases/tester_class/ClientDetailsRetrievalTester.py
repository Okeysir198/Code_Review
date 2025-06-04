import time
import json
from typing import List, Dict, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class ClientDetailsRetrievalTester:
    """A test harness for evaluating the client details retrieval tool."""
    
    def __init__(self, retrieve_client_details: Any, db_cartrack: Any):
        """Initialize the tester."""
        self.test_results = []
        self.retrieve_client_details = retrieve_client_details
        self.db_cartrack = db_cartrack

        
    def run_test(self, user_id: int, expected_attributes: Optional[Dict[str, Any]] = None, 
             expected_status: str = "SUCCESS"):
        """
        Run a single test case for the client details retrieval tool.
        
        Args:
            user_id: The user ID to test retrieval for
            expected_attributes: Dictionary of expected attributes and their values (optional)
            expected_status: Expected result status - SUCCESS, NOT_FOUND, ERROR
        
        Returns:
            The test result dictionary
        """
        # Record test case details
        test_case = {
            "user_id": user_id,
            "expected_attributes": expected_attributes,
            "expected_status": expected_status
        }
        
        # Start timer
        start_time = time.time()
        
        # Run retrieval
        error_message = None  # Initialize error_message variable
        try:
            result = self.retrieve_client_details.invoke({"user_id": user_id, "db_cartrack": self.db_cartrack})
            # Determine status based on result
            if result is None:
                status = "NOT_FOUND"
            else:
                status = "SUCCESS"
        except Exception as e:
            result = None
            status = "ERROR"
            error_message = str(e)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Check if attributes match expected values (if provided)
        attributes_match = True
        attribute_failures = []
        
        if expected_attributes and result:
            for key, expected_value in expected_attributes.items():
                # Handle nested attributes with dot notation
                if "." in key:
                    parts = key.split(".")
                    actual_value = result
                    try:
                        for part in parts:
                            if part.isdigit() and isinstance(actual_value, list):
                                actual_value = actual_value[int(part)]
                            elif isinstance(actual_value, dict):
                                actual_value = actual_value.get(part)
                            else:
                                actual_value = None
                                break
                    except (IndexError, KeyError, TypeError):
                        actual_value = None
                else:
                    actual_value = result.get(key)
                
                if actual_value != expected_value:
                    attributes_match = False
                    attribute_failures.append({
                        "attribute": key,
                        "expected": expected_value,
                        "actual": actual_value
                    })
        
        # Check if status matches expected status
        status_match = (status == expected_status)
        
        # Overall pass/fail
        passed = status_match and attributes_match
        
        # Store complete test results
        test_result = {
            "test_case": test_case,
            "result": result,
            "status": status,
            "execution_time": execution_time,
            "passed": passed,
            "status_match": status_match,
            "attributes_match": attributes_match,
            "attribute_failures": attribute_failures,
            "error_message": error_message if status == "ERROR" else None
        }
        self.test_results.append(test_result)
        
        # Print test result summary
        self._print_test_result(test_result)
        
        return test_result
    
    def run_all_tests(self, test_cases):
        """Run all predefined test cases."""
        print(f"\n{COLORS['HEADER']}{COLORS['BOLD']}==== STARTING CLIENT DETAILS RETRIEVAL TEST SUITE ===={COLORS['END']}\n")
        
        # Run all test cases
        for case in test_cases:
            self.run_test(
                user_id=case["user_id"],
                expected_attributes=case.get("expected_attributes"),
                expected_status=case.get("expected_status", "SUCCESS")
            )
        
        # Print summary
        self._print_summary()
    
    def _print_test_result(self, test_result):
        """Print a single test result in readable format."""
        print(f"\n{COLORS['HEADER']}{COLORS['BOLD']}----- TEST CASE: User ID {test_result['test_case']['user_id']} -----{COLORS['END']}")
        
        # Status with color based on result
        status_color = {
            "SUCCESS": COLORS['SUCCESS'],
            "NOT_FOUND": COLORS['WARNING'],
            "ERROR": COLORS['ERROR']
        }.get(test_result['status'], COLORS['INFO'])
        
        print(f"{COLORS['BOLD']}Status:{COLORS['END']} {status_color}{test_result['status']}{COLORS['END']}")
        
        if test_result['status'] == "ERROR":
            print(f"{COLORS['BOLD']}Error:{COLORS['END']} {COLORS['ERROR']}{test_result['error_message']}{COLORS['END']}")
        
        print(f"{COLORS['BOLD']}Execution time:{COLORS['END']} {COLORS['INFO']}{test_result['execution_time']:.3f} seconds{COLORS['END']}")
        
        # Print expected status
        expected_status = test_result['test_case']['expected_status']
        if test_result['status_match']:
            status_match_text = f"{COLORS['SUCCESS']}✓ MATCHES{COLORS['END']}"
        else:
            status_match_text = f"{COLORS['ERROR']}✗ MISMATCH (expected {expected_status}){COLORS['END']}"
        print(f"{COLORS['BOLD']}Status check:{COLORS['END']} {status_match_text}")
        
        # Print attribute validation results if applicable
        if test_result['test_case']['expected_attributes']:
            if test_result['attributes_match']:
                attr_match_text = f"{COLORS['SUCCESS']}✓ ALL MATCH{COLORS['END']}"
            else:
                attr_match_text = f"{COLORS['ERROR']}✗ {len(test_result['attribute_failures'])} MISMATCHES{COLORS['END']}"
            print(f"{COLORS['BOLD']}Attribute check:{COLORS['END']} {attr_match_text}")
            
            # Print attribute failures if any
            if not test_result['attributes_match']:
                print(f"\n{COLORS['BOLD']}Attribute failures:{COLORS['END']}")
                for failure in test_result['attribute_failures']:
                    print(f"  {COLORS['ERROR']}• {failure['attribute']}:{COLORS['END']}")
                    print(f"    Expected: {COLORS['INFO']}{failure['expected']}{COLORS['END']}")
                    print(f"    Actual: {COLORS['WARNING']}{failure['actual']}{COLORS['END']}")
        
        # Print result data preview if available
        if test_result['result']:
            print(f"\n{COLORS['BOLD']}Result preview:{COLORS['END']}")
            result_preview = {k: test_result['result'][k] for k in list(test_result['result'].keys())[:3]}
            print(f"  {COLORS['INFO']}{json.dumps(result_preview, indent=2)[:200]}...{COLORS['END']}")
            
            # Print vehicle count if available
            if 'vehicles' in test_result['result'] and isinstance(test_result['result']['vehicles'], list):
                vehicle_count = len(test_result['result']['vehicles'])
                print(f"\n{COLORS['BOLD']}Vehicle count:{COLORS['END']} {COLORS['INFO']}{vehicle_count}{COLORS['END']}")
    
    def _print_summary(self):
        """Print summary of all test results."""
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r['passed'])
        
        print(f"\n{COLORS['HEADER']}{COLORS['BOLD']}==== TEST SUMMARY ===={COLORS['END']}")
        print(f"{COLORS['BOLD']}Total tests:{COLORS['END']} {COLORS['INFO']}{total}{COLORS['END']}")
        
        # Color the pass/fail numbers based on count
        pass_color = COLORS['SUCCESS'] if passed == total else COLORS['WARNING']
        fail_color = COLORS['SUCCESS'] if (total - passed) == 0 else COLORS['ERROR']
        
        print(f"{COLORS['BOLD']}Passed:{COLORS['END']} {pass_color}{passed}{COLORS['END']}")
        print(f"{COLORS['BOLD']}Failed:{COLORS['END']} {fail_color}{total - passed}{COLORS['END']}")
        
        # Average execution time
        avg_time = sum(r['execution_time'] for r in self.test_results) / total
        print(f"{COLORS['BOLD']}Average execution time:{COLORS['END']} {COLORS['INFO']}{avg_time:.3f} seconds{COLORS['END']}")
        
        # Result distribution
        status_counts = {}
        for r in self.test_results:
            status = r['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\n{COLORS['BOLD']}Status distribution:{COLORS['END']}")
        
        # Color code different status types
        status_colors = {
            "SUCCESS": COLORS['SUCCESS'],
            "NOT_FOUND": COLORS['WARNING'],
            "ERROR": COLORS['ERROR']
        }
        
        for status, count in status_counts.items():
            color = status_colors.get(status, COLORS['INFO'])
            print(f"  {color}{status}:{COLORS['END']} {count}")
    
    def export_results(self, filename="client_details_test_results.json"):
        """Export test results to JSON file."""
        # Create a serializable version of results
        serializable_results = []
        for result in self.test_results:
            # Make a copy to avoid modifying the original
            serializable_result = result.copy()
            
            # Handle any non-serializable data in the result
            if 'result' in serializable_result and serializable_result['result'] is not None:
                try:
                    # Test if it's serializable
                    json.dumps(serializable_result['result'])
                except (TypeError, OverflowError):
                    # If not serializable, convert to string representation
                    serializable_result['result'] = str(serializable_result['result'])
            
            serializable_results.append(serializable_result)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n{COLORS['SUCCESS']}Test results exported to {COLORS['BOLD']}{filename}{COLORS['END']}")

#########################################################################################
# Test cases
#########################################################################################

test_cases = [
    {
        "user_id": 107620,
        "expected_status": "SUCCESS",
        "expected_attributes": {
            "full_name": "Sue-Carla Ruyter",  # Replace with actual expected name
            "email": "dev@onecell.co.za"  # Replace with actual expected email
        }
    },
    {
        "user_id": 107621,
        "expected_status": "SUCCESS",
        "expected_attributes": {
            "vehicles.0.make": "Audi",  # Checks first vehicle's make
            "vehicles.0.model": "Q3"  # Checks first vehicle's model
        }
    },
    {
        "user_id": 9999232399,  # Likely non-existent user
        "expected_status": "NOT_FOUND"
    },
    # Add more test cases as needed
]
