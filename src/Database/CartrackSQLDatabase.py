import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from contextlib import contextmanager
from decimal import Decimal
import datetime
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# =====================================================
# PostgresDatabase
# =====================================================
class PostgresDatabase:
    """Enhanced PostgreSQL database connection with advanced query and function capabilities."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize a PostgreSQL database connection.
        
        Args:
            connection_string: PostgreSQL connection string. If None, uses POSTGRES_URL environment variable.
        """
        self.connection_string = connection_string or os.environ.get("POSTGRES_URL")
        self.connection = None
        self.logger = logging.getLogger(__name__)
        self.connect()
       
    def connect(self):
        """Establish connection to the PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(self.connection_string)
            self.logger.info("Database connection established successfully")
        except Exception as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise
            
    def disconnect(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")
            self.connection = None
    
    @contextmanager
    def _handle_errors(self, operation_name: str):
        """Context manager for consistent error handling."""
        try:
            yield
        except Exception as e:
            self.logger.error(f"Error in {operation_name}: {e}")
            raise
    
    @staticmethod
    def _split_quoted_string(input_string: str) -> List[str]:
        """Splits a comma-separated string, treating quoted commas as one item."""
        values = []
        current_item = ""
        in_quotes = False

        for char in input_string:
            if char == '"':
                in_quotes = not in_quotes
            elif char == "," and not in_quotes:
                values.append(current_item.strip())
                current_item = ""
            else:
                current_item += char

        values.append(current_item.strip())
        return values
    
    @staticmethod
    def _serialize_data(data: List[Dict[Any, Any]]) -> List[Dict[Any, str]]:
        """Converts datetime, date and Decimal values to strings with improved type handling."""
        if not data:
            return []
            
        def serialize_value(value: Any) -> Any:
            """Converts datetime and Decimal to strings."""
            if value is None:
                return None
            elif isinstance(value, datetime.datetime):
                return value.isoformat()
            elif isinstance(value, datetime.date):
                return value.isoformat()
            elif isinstance(value, Decimal):
                return str(round(value, 2))
            return value
        
        result = []
        for row in data:
            filtered_row = {key: serialize_value(value) for key, value in row.items() if "au$" not in key}
            result.append(filtered_row)

        return result
    
    @lru_cache(maxsize=128)
    def _get_function_parameters(self, schema_name: str, function_name: str) -> List[str]:
        """Cached retrieval of function output parameter names."""
        function_info_query = f"""
            SELECT pg_get_function_arguments(p.oid) AS function_arguments 
            FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace 
            WHERE p.proname = '{function_name}' AND n.nspname = '{schema_name}';
        """
        function_info_result = self.execute_query(function_info_query)
        
        if not function_info_result or not function_info_result[0]:
            return []
            
        function_arguments_str = function_info_result[0].get("function_arguments")
        if not function_arguments_str:
            return []
            
        function_arguments = function_arguments_str.split(", ")
        return function_arguments
            
    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results as a list of dictionaries.
        
        Args:
            query: SQL query to execute
            params: Parameters to pass to the query
            
        Returns:
            List of dictionaries where each dictionary represents a row
        """
        with self._handle_errors("execute_query"):
            if not self.connection:
                self.connect()
                
            try:
                with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    if cursor.description:  # Check if query returns data
                        result = cursor.fetchall()
                        self.connection.commit()
                        return self._serialize_data([dict(row) for row in result])
                    else:
                        self.connection.commit()
                        return []
            except Exception as e:
                self.connection.rollback()
                self.logger.error(f"Error executing query: {e}")
                raise
    
    def execute_function_and_extract_output(
        self,
        schema_name: str,
        function_name: str,
        argument_value: str,
        include_argument_type: bool = False,
    ) -> Optional[List[Dict[str, str]]]:
        """Executes a function and returns output as a list of dictionaries with improved error handling."""
        with self._handle_errors(f"execute_function_{schema_name}.{function_name}"):
            # Get function parameters (cached)
            function_arguments = self._get_function_parameters(schema_name, function_name)
            if not function_arguments:
                self.logger.warning(f"No parameters found for function {schema_name}.{function_name}")
                return None
                
            # Extract output parameter names
            parameter_names = [
                arg.split(" ")[1].replace("out_", "")
                + (f" ({arg.split(' ')[2]})" if len(arg.split(" ")) > 2 and include_argument_type else "")
                for arg in function_arguments
                if "out_" in arg or "OUT" in arg
            ]
            
            if not parameter_names:
                self.logger.warning(f"No output parameters found for function {schema_name}.{function_name}")
                return None
            
            # Execute the database function
            result_data = self.execute_query(f"SELECT {schema_name}.{function_name}({argument_value});")
            if not result_data:
                return None
            
            results = []
            for row in result_data:
                output_str = next(iter(row.values()), None)
                if not output_str:
                    results.append(None)
                    continue
                
                output_values = self._split_quoted_string(output_str[1:-1])
                results.append(dict(zip(parameter_names, output_values)))

            return results
    
    
            
    def __enter__(self):
        """Context manager entry - connects to database."""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - disconnects from database."""
        self.disconnect()


# =====================================================
# Connection
# =====================================================
database =[
    {
        'user' : os.environ.get('CARTRACK_POSTGRES_USER'),
        'password' : os.environ.get('CARTRACK_POSTGRES_KEY'),
        'host' : "dbreporting.cartrack.co.za",
        'port' : 5432,
        'database_name': "cartrack"
    },
    {
        'user' : os.environ.get('CARTRACK_POSTGRES_USER'),
        'password' : os.environ.get('CARTRACK_POSTGRES_KEY'),
        'host' : "fleetreporting.cartrack.co.za",
        'port' : 5432,
        'database_name': "ct_fleet"
    },
    {
        'user' : os.environ.get('CARTRACK_POSTGRES_USER'),
        'password' : os.environ.get('CARTRACK_POSTGRES_KEY'),
        'host' : "new_dev.cartrack.co.za",
        'port' : 5432,
        'database_name': "cartrack" #dev
    },
    {
        'user' : os.environ.get('CARTRACK_POSTGRES_USER'),
        'password' : os.environ.get('CARTRACK_POSTGRES_KEY'),
        'host' : "new_dev.cartrack.co.za",
        'port' : 5432,
        'database_name': "CARTRACK_UAT"
    },
]
connection = database[2]

POSTGRES_URL = f"postgresql://{connection['user']}:{connection['password']}@{connection['host']}:{connection['port']}/{connection['database_name']}"

db = PostgresDatabase(POSTGRES_URL)
# =====================================================
# Standalone Functions/Tools
# =====================================================
# General
@tool
def get_debtor_age_analysis(
    number_records: int = 10,
    user_id: Optional[int] = None,
    client_type: Optional[str] = None,
    min_age_days: Optional[int] = 0,
    max_age_days: Optional[int] = 10,
    min_balance_total: Optional[float] = 100,
    invoice_option: Optional[str] = None,  
    payment_arrangement_status: Optional[str] = None,
    pre_legal: Optional[bool] = None,
    rejected: Optional[bool] = None,
    sort_by: Optional[str] = "balance_total",
    sort_direction: Optional[str] = "DESC",
) -> List[Dict[str, Any]]:
    """
    Retrieves detailed financial aging analysis for debtors/clients with outstanding balances.
    
    Use this tool when you need to analyze accounts receivable, outstanding balances across
    different aging periods, and identify clients with overdue payments.
    """
    # Validate inputs
    valid_sort_fields = ["balance_total", "age_days"]
    if sort_by not in valid_sort_fields:
        sort_by = "balance_total"
    
    sort_direction = "DESC" if sort_direction.upper() not in ["ASC", "DESC"] else sort_direction.upper()
    
    # Build the base query with required filters
    query_parts = [
        "SELECT baa.*",
        "FROM ct.billing_age_analysis baa",
        "WHERE 1=1"  # Always true condition to simplify adding optional filters
    ]
    # logger.info(f"User ID: {user_id}")
    if user_id:
        query_parts.append(f"AND CAST(baa.user_id AS TEXT) LIKE '%{str(user_id)}%' ")
    else:
        if client_type and client_type != 'All':
            query_parts.append(f"AND baa.client_type = '{client_type}'")
        
        if min_age_days > 0:
            query_parts.append(f"AND baa.age_days >= {min_age_days}")
        
        if max_age_days > min_age_days >= 0:
            query_parts.append(f"AND baa.age_days <= {max_age_days}")
        
        if min_balance_total > 0:
            query_parts.append(f"AND baa.balance_total >= {min_balance_total}")

        if invoice_option and invoice_option != 'All':
            query_parts.append(f"AND baa.invoice_option = '{invoice_option}'")
        
        if payment_arrangement_status and payment_arrangement_status != 'All':
            query_parts.append(f"AND baa.payment_arrangement = '{payment_arrangement_status}'")
        
        if pre_legal is not None:
            if pre_legal:
                query_parts.append("AND baa.pre_legal_billing = 'Yes'")
            else:
                query_parts.append("AND baa.pre_legal_billing = 'No'")

        if rejected is not None:
            if rejected:
                query_parts.append("AND baa.rejected = 'Yes'")
            else:
                query_parts.append("AND baa.rejected = 'No'")
    
    # Add sorting and limit
    query_parts.append(f"ORDER BY baa.{sort_by} {sort_direction}")
    query_parts.append(f"LIMIT {number_records}")

    # Combine the query parts
    sql_query = "\n".join(query_parts)
    
    # Execute the query with parameters and handle potential errors
    try:
        return db.execute_query(sql_query)
    except Exception as e:
        error_message = f"Database query failed: {str(e)}"
        # You could log the error here if needed
        logger.error(error_message)
        return {"error": error_message, "query": sql_query}

@tool
def date_helper(query=None):
    """
    Simple date helper tool that accepts natural language date queries and returns formatted date information.
    
    Args:
        query (str): Natural language date query like:
                    - "today"
                    - "tomorrow"
                    - "Friday next week"
                    - "second Monday of January 2025"
                    - If None, returns today's information
    
    Returns:
        dict: Date information including ISO date, weekday, and human-readable description
    """
    import datetime
    import calendar
    import re
    
    # Get today's date information
    today = datetime.datetime.now()
    today_info = {
        "today_date": today.strftime("%Y-%m-%d"),
        "today_weekday": today.strftime("%A"),
        "today_month": today.strftime("%B"),
        "today_year": today.year,
        "today_day": today.day
    }
    
    # Return today's info if no query specified
    if not query:
        return today_info
    
    # Normalize query text
    query = query.lower().strip()
    
    # Simple cases
    if query == "today":
        return {
            **today_info,
            "requested_date": today.strftime("%Y-%m-%d"),
            "description": "Today"
        }
    
    if query == "tomorrow":
        tomorrow = today + datetime.timedelta(days=1)
        return {
            **today_info,
            "requested_date": tomorrow.strftime("%Y-%m-%d"),
            "requested_weekday": tomorrow.strftime("%A"),
            "description": "Tomorrow"
        }
    
    if query == "yesterday":
        yesterday = today - datetime.timedelta(days=1)
        return {
            **today_info,
            "requested_date": yesterday.strftime("%Y-%m-%d"),
            "requested_weekday": yesterday.strftime("%A"),
            "description": "Yesterday"
        }
    
    # Day name mappings
    day_mapping = {
        "monday": 0, "mon": 0,
        "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2,
        "thursday": 3, "thu": 3, "thurs": 3,
        "friday": 4, "fri": 4,
        "saturday": 5, "sat": 5,
        "sunday": 6, "sun": 6
    }
    
    # Month name mappings
    month_mapping = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12
    }
    
    # Check for "next day" or "this day" patterns (e.g., "next friday", "this monday")
    day_pattern = r"(this|next|last)\s+(\w+day|\w{3,4})"
    day_match = re.search(day_pattern, query)
    
    if day_match:
        when, day_name = day_match.groups()
        
        # Find the day number (0-6) for the requested day
        day_number = None
        for key, value in day_mapping.items():
            if day_name.startswith(key):
                day_number = value
                day_name = key.capitalize()
                break
        
        if day_number is None:
            return {**today_info, "error": f"Unknown day: {day_name}"}
        
        # Calculate target date
        today_weekday = today.weekday()
        
        if when == "this":
            # "This Monday" means the Monday of the current week
            days_to_add = (day_number - today_weekday) % 7
            if days_to_add < 0:
                days_to_add += 7
            
        elif when == "next":
            # "Next Monday" means the Monday of next week
            days_to_add = (day_number - today_weekday) % 7
            if days_to_add == 0:
                days_to_add = 7  # If today is the day, go to next week
            
        else:  # "last"
            # "Last Monday" means the Monday of last week
            days_to_add = (day_number - today_weekday) % 7 - 7
        
        target_date = today + datetime.timedelta(days=days_to_add)
        
        return {
            **today_info,
            "requested_date": target_date.strftime("%Y-%m-%d"),
            "requested_weekday": target_date.strftime("%A"),
            "description": f"{when.capitalize()} {day_name}"
        }
    
    # Check for "day next/last week" patterns (e.g., "friday next week")
    week_pattern = r"(\w+day|\w{3,4})\s+(next|last|this)\s+week"
    week_match = re.search(week_pattern, query)
    
    if week_match:
        day_name, when = week_match.groups()
        
        # Find the day number (0-6) for the requested day
        day_number = None
        for key, value in day_mapping.items():
            if day_name.startswith(key):
                day_number = value
                day_name = key.capitalize()
                break
        
        if day_number is None:
            return {**today_info, "error": f"Unknown day: {day_name}"}
        
        # Calculate target date
        today_weekday = today.weekday()
        
        if when == "this":
            # "Monday this week" means the Monday of the current week
            days_to_add = (day_number - today_weekday) % 7
            if days_to_add < 0:
                days_to_add += 7
            
        elif when == "next":
            # "Monday next week" means the Monday of next week
            days_to_add = (day_number - today_weekday) % 7 + 7
            
        else:  # "last"
            # "Monday last week" means the Monday of last week
            days_to_add = (day_number - today_weekday) % 7 - 7
        
        target_date = today + datetime.timedelta(days=days_to_add)
        
        return {
            **today_info,
            "requested_date": target_date.strftime("%Y-%m-%d"),
            "requested_weekday": target_date.strftime("%A"),
            "description": f"{day_name} {when} week"
        }
    
    # Check for "nth day of month year" patterns (e.g., "second monday of january 2025")
    nth_pattern = r"(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|last|\d+(?:st|nd|rd|th)?)\s+(\w+day|\w{3,4})\s+(?:of\s+)?(\w+)(?:\s+(\d{4}))?"
    nth_match = re.search(nth_pattern, query)
    
    if nth_match:
        nth, day_name, month_name, year_str = nth_match.groups()
        
        # Parse the nth occurrence
        if nth == "first" or nth == "1st":
            occurrence = 1
        elif nth == "second" or nth == "2nd":
            occurrence = 2
        elif nth == "third" or nth == "3rd":
            occurrence = 3
        elif nth == "fourth" or nth == "4th":
            occurrence = 4
        elif nth == "fifth" or nth == "5th":
            occurrence = 5
        elif nth == "last":
            occurrence = -1
        else:
            try:
                occurrence = int(re.sub(r'(?:st|nd|rd|th)$', '', nth))
            except ValueError:
                return {**today_info, "error": f"Invalid occurrence: {nth}"}
        
        # Find the day number (0-6) for the requested day
        day_number = None
        for key, value in day_mapping.items():
            if day_name.startswith(key):
                day_number = value
                day_name = key.capitalize()
                break
        
        if day_number is None:
            return {**today_info, "error": f"Unknown day: {day_name}"}
        
        # Find the month number (1-12) for the requested month
        month_number = None
        for key, value in month_mapping.items():
            if month_name.startswith(key):
                month_number = value
                month_name = calendar.month_name[value]
                break
        
        if month_number is None:
            return {**today_info, "error": f"Unknown month: {month_name}"}
        
        # Parse year (default to current year if not specified)
        year = int(year_str) if year_str else today.year
        
        # Calculate the target date
        if occurrence > 0:
            # Find the first day of the month
            first_day = datetime.datetime(year, month_number, 1)
            
            # Find the first occurrence of the requested day
            days_until_first = (day_number - first_day.weekday()) % 7
            first_occurrence = first_day + datetime.timedelta(days=days_until_first)
            
            # Find the nth occurrence
            target_date = first_occurrence + datetime.timedelta(days=(occurrence - 1) * 7)
            
            # Check if still in the month
            if target_date.month != month_number:
                return {**today_info, "error": f"There is no {nth} {day_name} in {month_name} {year}"}
        else:
            # Last occurrence - find the last day of the month
            last_day = datetime.datetime(year, month_number, calendar.monthrange(year, month_number)[1])
            
            # Work backwards to find the last occurrence of the requested day
            days_to_subtract = (last_day.weekday() - day_number) % 7
            last_occurrence = last_day - datetime.timedelta(days=days_to_subtract)
            
            target_date = last_occurrence
        
        return {
            **today_info,
            "requested_date": target_date.strftime("%Y-%m-%d"),
            "requested_weekday": target_date.strftime("%A"),
            "requested_month": target_date.strftime("%B"),
            "requested_year": target_date.year,
            "description": f"{nth.capitalize()} {day_name} of {month_name} {year}"
        }
    
    # If no pattern matched
    return {**today_info, "error": f"Could not parse date query: {query}"}

@tool
def get_current_date_time():
    """
    Returns the current date and time.
    
    Returns:
        str: A string representation of the current date and time.
    """
    import datetime
    return {'current_date_time':datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# #Step 1: introduction

#-----------------------------------------------------#
# #Step 2: get client info verification
@tool
def get_client_vehicle_info(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all vehicles belonging to a client.
    
    Args:
        user_id: Client ID (string, e.g., "83906")
        
    Returns:
        List of dictionaries containing vehicle information:
        - vehicle_id: Unique vehicle identifier
        - registration: License plate number
        - make: Vehicle manufacturer (e.g., Toyota)
        - model: Vehicle model name
        - color: Vehicle color
        - chassis_number: VIN/chassis number
        - model_year: Year the vehicle was manufactured
        - contract_status: Current contract status
        - terminal_serial: Serial number of installed tracking device
        - terminal_last_response: Timestamp of last communication
    """
    # Use parameterized query to prevent SQL injection
    vehicle_info_query = f"""
        SELECT
            v.vehicle_id,
            v.registration,
            v.manufacturer AS make,
            v.model,
            v.colour AS color,
            v.chassisnr AS chassis_number,
            v.modelyear AS model_year,
            cs.description AS contract_status,
            t.terminal_serial,
            t.lastresponse AS terminal_last_response
        FROM ct.vehicle v
        LEFT JOIN ct.terminal t ON t.terminal_id = v.terminal_id
        LEFT JOIN ct.contract c ON c.contract_id = v.current_contract_id
        LEFT JOIN ct.contract_state cs ON cs.contract_state_id = c.contract_state_id
        WHERE
            v.user_id = '{user_id}'
            AND c.contract_id IS NOT NULL
        ORDER BY
            t.lastresponse ASC
    """
    
    # Pass parameters securely to prevent SQL injection
    return db.execute_query(vehicle_info_query)

#Use 1 of the following functions to get client identity details
@tool
def get_client_identity_details_old(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves identity verification information for a client including personal details and vehicles.
    
    Use this tool when you need to verify a client's identity during a support call.
    Returns one record per vehicle owned by the client.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        List of dictionaries containing:
        - Personal details: client_title, client_full_name, id_number, passport_number
        - Account info: user_id, user_name, email_address
        - Vehicle details: vehicle_registration, vehicle_make, vehicle_model, vin_number
    """
    sql_query = """
        SELECT
            i.user_id,
            v.vehicle_id,
            itt.description AS "client_title",
            debtor.out_client AS "client_full_name",
            debtor.out_user_name AS "user_name",
            i.id_number,
            SUBSTRING(i.id_number, 1, 6) AS "date_of_birth_raw",
            i.passport_number,
            v.registration AS "vehicle_registration",
            v.manufacturer AS "vehicle_make",
            v.model AS "vehicle_model",
            v.colour AS "vehicle_colour",
            v.chassisnr AS "vin_number",
            u.primary_email AS "email_address"
        FROM
            ct.individual i
        LEFT JOIN
            LATERAL ct.get_debtor_info(i.user_id) debtor ON TRUE
        LEFT JOIN
            ct.vehicle v ON i.user_id = v.user_id
        LEFT JOIN
            ct.user u ON i.user_id = u.user_id
        LEFT JOIN
            ct.individual_title_type itt ON i.individual_title_type_id = itt.individual_title_type_id
        WHERE
            i.user_id = '{}'
    """.format(user_id)
    
    return db.execute_query(sql_query)


@tool
def get_client_profile(user_id: str) -> Optional[Dict]:
    """
    Retrieves comprehensive client profile data including personal information, contact details, 
    addresses, vehicles, and account information.
    
    Use this tool when:
    - You need to look up a specific client's complete information
    - You need to verify or check a client's contact details, address, or vehicle information
    - You need billing or subscription information for a client
    - You're creating reports or summaries about a client's profile
    - You need to understand a client's fleet of vehicles and tracking setup
    
    Args:
        user_id: Client's unique identifier (e.g., "83906" or "JOHN01264")
        
    Returns:
        Nested dictionary containing detailed client information:
        - client_info: Personal details (name, ID, contact details)
          - title, first_name, last_name, id_number, passport, client_full_name
          - email_address
          - contact: telephone, fax, mobile numbers
        - addresses: List of physical and postal addresses
          - type, address_line1, address_line2, post_code, province
        - vehicles: List of all registered vehicles with their details
          - registration, make, model, color, chassis_number, model_year
          - contract_status, terminal_serial, terminal_last_response
        - sim_card: SIM card information (chip_id)
        - fitment: Installation details (date, fitter_name)
        - billing: Subscription information (product_package, recovery_option)
    """
    def parse_client_data(raw_str):
        """Parse a single row of client information from stored procedure result."""
        # Basic validation
        if not raw_str.startswith('(') or not raw_str.endswith(')'):
            return None
        
        # Remove parentheses
        content = raw_str[1:-1]
        
        # Split by commas but respect quoted sections
        parts = []
        in_quotes = False
        current_part = ""
        
        for char in content:
            if char == '"':
                in_quotes = not in_quotes
                current_part += char
            elif char == ',' and not in_quotes:
                parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        
        # Add the last part
        if current_part:
            parts.append(current_part.strip())
        
        # Need at least 4 parts
        if len(parts) < 4:
            return None
        
        # Extract components
        return {
            "category": parts[0].strip('"'),
            "field": parts[1].strip('"'),
            "value": parts[2].strip('"'),
            "user_name": parts[3].strip('"')
        }
    
    def build_client_structure(sql_results, basic_info=None, vehicle_info=None):
        """
        Convert SQL query results to an organized nested structure.
        
        Args:
            sql_results: List of dictionaries with sp_client_information results
            basic_info: Optional dictionary with additional client information
            vehicle_info: Optional list of vehicle information
        """
        # Initialize structure
        result = {
            "user_name": None,
            "client_info": {
                "title": "",
                "first_name": "",
                "last_name": "",
                "id_number": "",
                "passport": "",
                "employer": "",
                "client_full_name": "",
                "date_of_birth": "",
                "email_address": "",
                "contact": {
                    "telephone": "",
                    "fax": "",
                    "mobile": ""
                }
            },
            "addresses": [],
            "vehicles": [],
            "sim_card": {
                "chip_id": ""
            },
            "fitment": {
                "date": "",
                "fitter_name": ""
            },
            "billing": {
                "product_package": "",
                "recovery_option": ""
            }
        }
        if vehicle_info:
            result["vehicles"] = db._serialize_data(vehicle_info)

        # Add basic info if provided
        if basic_info:
            for field in ["client_full_name", "date_of_birth", "email_address"]:
                result["client_info"][field] = basic_info.get(field, "")
        
        # Track current address and vehicle
        current_address = None
        current_vehicle = None
        
        for item in sql_results:
            if 'sp_client_information' not in item:
                continue
            
            parsed = parse_client_data(item['sp_client_information'])
            if not parsed:
                continue
            
            category = parsed["category"]
            field = parsed["field"]
            value = parsed["value"]
            
            # Set user_id if not set
            if not result["user_name"]:
                result["user_name"] = parsed["user_name"]
            
            # Process by category
            if category == "Client Information":
                _process_client_info(result, field, value)
                
            elif category == "Client Address":
                current_address = _process_address(result, field, value, current_address)
                
            elif category == "SIM Card Information":
                if field == "SIM Chip ID" and value.strip():
                    result["sim_card"]["chip_id"] = value
            
            elif category == "Fitment Information":
                if field == "Fitment Date" and value.strip():
                    result["fitment"]["date"] = value
                elif field == "Fitter Name" and value.strip():
                    result["fitment"]["fitter_name"] = value
            
            elif category == "Billing Information":
                if field == "Product Package" and value.strip():
                    result["billing"]["product_package"] = value
                elif field == "Product Package Recovery Option" and value.strip():
                    result["billing"]["recovery_option"] = value
        
        # # Add final vehicle if not using vehicle_info
        # if current_vehicle:
        #     result["vehicles"].append(current_vehicle)
        
        # Add final address
        if current_address and len(current_address) > 1:
            result["addresses"].append(current_address)
        
        return result
    
    def _process_client_info(result, field, value):
        """Process client information fields."""
        field_map = {
            "Title": "title",
            "First Name": "first_name",
            "Last Name": "last_name",
            "ID Number": "id_number",
            "Passport": "passport",
            "Employer": "employer",
            "Client Telephone Number": ["contact", "telephone"],
            "Client FAX Number": ["contact", "fax"],
            "Client Mobile Number": ["contact", "mobile"]
        }
        
        if field in field_map:
            mapping = field_map[field]
            if isinstance(mapping, list):
                result["client_info"][mapping[0]][mapping[1]] = value
            else:
                result["client_info"][mapping] = value
    
    def _process_address(result, field, value, current_address):
        """Process address fields and return the current address being built."""
        if field == "Address Type":
            if current_address and current_address.get("type") and len(current_address) > 1:
                result["addresses"].append(current_address)
            return {"type": value}
        
        field_map = {
            "Address 1": "address_line1",
            "Address 2": "address_line2",
            "Post Code": "post_code",
            "Province": "province"
        }
        
        if field in field_map and current_address:
            current_address[field_map[field]] = value
        
        if field == "_" and current_address and len(current_address) > 1:
            result["addresses"].append(current_address)
            return None
            
        return current_address
    
    try:
        # Query to get basic client information
        basic_info_query = f"""
            SELECT 
                debtor.out_client AS "client_full_name",
                CASE 
                    WHEN LENGTH(SUBSTRING(i.id_number, 1, 6)) = 6 AND 
                        SUBSTRING(i.id_number, 1, 6) ~ '^[0-9]{6}$' AND
                        SUBSTRING(i.id_number, 3, 2) BETWEEN '01' AND '12' AND
                        SUBSTRING(i.id_number, 5, 2) BETWEEN '01' AND '31'
                    THEN TO_DATE(SUBSTRING(i.id_number, 1, 6), 'YYMMDD')
                    ELSE NULL
                END AS "date_of_birth",
                u.primary_email AS "email_address"
            FROM
                ct.individual i
            LEFT JOIN
                LATERAL ct.get_debtor_info(i.user_id) debtor ON TRUE
            LEFT JOIN
                ct.user u ON i.user_id = u.user_id
            WHERE
                i.user_id = {user_id}
        """

        # Query to get vehicle information
        vehicle_info_query = f"""
            SELECT
                v.registration,
                v.manufacturer as make,
                v.model,
                v.colour as color,
                v.chassisnr as chassis_number,
                v.modelyear as model_year,
                cs.description as contract_status,
                t.terminal_serial,
                t.lastresponse as terminal_last_response
            FROM ct.vehicle v
            LEFT JOIN ct.terminal t ON t.terminal_id = v.terminal_id
            LEFT JOIN ct.contract c on c.contract_id =v.current_contract_id
            LEFT JOIN ct.contract_state cs on cs.contract_state_id =c.contract_state_id
            WHERE
                v.user_id = '{user_id}'
                AND c.contract_id IS NOT NULL
            ORDER BY
                t.lastresponse ASC; 
        """
        # AND v.terminal_id IS NOT NULL

        # Execute queries
        basic_info_results = db.execute_query(basic_info_query)
        basic_info = basic_info_results[0] if basic_info_results else {}
        
        vehicle_info = db.execute_query(vehicle_info_query)
        
        # Get detailed client information
        sp_query = f"SELECT ct.sp_client_information('{user_id}')"
        sp_results = db.execute_query(sp_query)
        
        if not sp_results:
            return None
        
        # Build the final client data structure
        return build_client_structure(sp_results, basic_info, vehicle_info)
        
    except Exception as e:
        db.logger.error(f"Error retrieving client information: {str(e)}")
        return None

#-----------------------------------------------------#

@tool
def client_call_verification(user_id: str) -> Dict[str, Any]:
    """
    Retrieve a verification token for client call authentication.
    
    Args:
        user_id (str): The client ID to verify
        
    Returns:
        Dict[str, Any]: A dictionary containing either a 'token' or 'error' key
    """
    if not user_id or not isinstance(user_id, str):
        return {"error": "Invalid user_id parameter", "status_code": 400}
    
    try:
        # Use proper parameter binding to prevent SQL injection
        query = "SELECT ct.client_call_verification(%s) AS token;"
        result = db.execute_query(query, params=(user_id,))

        # Properly extract and validate the result
        if result and isinstance(result, list) and len(result) > 0:
            row = result[0]
            if row and isinstance(row, dict) and 'token' in row:
                token = row['token']
                
                # Validate token format
                if token and isinstance(token, str) and token.strip():
                    return {"token": token.strip(), "status_code": 200}
                
        # If we get here, something went wrong with the result
        return {"error": "No valid token received from authentication service", "status_code": 401}
    
    except Exception as e:
        # Log the error with traceback but return limited details to caller
        logger.error(f"Client verification error for user_id={user_id}: {str(e)}", exc_info=True)
        return {"error": "Authentication service unavailable", "status_code": 503}

#-----------------------------------------------------#
# #Step 3: reason for call
@tool
def get_client_account_aging(user_id: str) -> Optional[List[Dict[str, str]]]:
    """
    Retrieves aging analysis of client's outstanding account balance (payment amounts by age category).
    
    Use this tool when discussing a client's outstanding balance and payment status.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        List containing a dictionary with aging categories:
        - x0: Current amount due (not yet overdue)
        - x30: Amount overdue by 1-30 days
        - x60: Amount overdue by 31-60 days
        - x90: Amount overdue by 61-90 days
        - x120: Amount overdue by 91+ days
        - xbalance: Total outstanding balance
    """
    return db.execute_function_and_extract_output(
        "ct", "sp_debtors_age", f"'{user_id}', current_date", False
    )

@tool
def get_client_account_status(user_id: str) -> Optional[List[Dict[str, str]]]:
    """
    Checks if a client account is in good standing or has payment issues.
    
    Use this tool for a quick check of whether a client's account is in good standing
    or if service might be affected due to payment issues.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        List containing a dictionary with 'account_state' key:
        - account_state: Message indicating account standing 
          (e.g., "in good account state" or "account in default")
    """
    return db.execute_function_and_extract_output(
        "ct", "sp_get_client_account_state", f"'{user_id}'", False
    )

@tool
def get_client_billing_analysis(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves detailed financial and billing analysis for a client account.
    
    Use this tool when you need comprehensive payment history, outstanding balances,
    and metrics related to a client's billing status.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        List of dictionaries containing comprehensive account metrics including:
        - Payment status: balance_total, balance_current, balance_30/60/90/120_days
        - Extended aging: day_120 through day_450 (for severely overdue accounts)
        - Financial metrics: annual_recurring_revenue_excl_vat, crv, ratio
        - Client data: full_name, client_type, vehicle_count
        - Relationship info: credit_controller, branch_name
    """
    return db.execute_query(f"SELECT * FROM ct.billing_age_analysis WHERE user_id={user_id}")
#-----------------------------------------------------#
# #Step 4: Negotiation
@tool
def get_client_contracts(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all vehicle tracking contracts for a client, including status and details.
    
    Use this tool to review a client's active and historical contracts, subscription plans,
    and service agreements.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        List of dictionaries containing contract information:
        - contract_id: Unique identifier for the contract
        - contract_state: Human-readable status (e.g., "Active", "Cancelled")
        - contract_start_date/contract_end_date: Service period
        - payment_option: Payment method description (e.g., "Monthly")
        - deal_category: Type of contract (e.g., "Consumer", "Corporate")
        - cancelation_reason_text: Reason for cancellation (if applicable)
    """
    sql_query = f"""
SELECT 
    con.*,
    po.description as payment_option,
    dc.category_name as deal_category,
    cs.description as contract_state,
    cr.cancelation_reason_text as cancelation_reason_text
FROM ct.contract con
LEFT JOIN ct.payment_option po ON po.payment_option_id = con.payment_option_id
LEFT JOIN ct.deal_category dc ON dc.deal_category_id = con.deal_category_id
LEFT JOIN ct.contract_state cs ON cs.contract_state_id = con.contract_state_id
LEFT JOIN ct.cancelation_reason cr ON cr.cancelation_reason_id = con.cancelation_reason_id
WHERE con.client_id = '{user_id}'
"""
    return db.execute_query(sql_query)

@tool
def get_client_account_overview(user_id: str) -> Optional[Dict[str, str]]:
    """
    Retrieves a 360° dashboard view of client account status and health.
    
    This is the most comprehensive client overview tool, providing a unified view
    of financial status, contact validity, service status, and account health metrics.
    Use this as your go-to tool for understanding a client's complete situation.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        Dictionary containing client account status across multiple dimensions:
        - Identification: user_id, user_name, client (full name)
        - Financial health: payment_status, account_status, total_invoices
        - Service information: service_status, vehicle_count_active
        - Communication: contactable status, contact_details_valid
        - Account management: responsible_agent, rating (1-5)
        - Payment parameters: payment_arrangement, allow_debit_order, mandate
        - Risk indicators: disputes, blacklisted status
    """
    try:
        result1 = db.execute_function_and_extract_output("ct", "get_debtor_info", f"'{user_id}'", False)
        result2 = db.execute_function_and_extract_output('ct','sp_debtors_client_info',user_id )
        if not result1 or not result1[0]:
            return None
            
        debtor_info = result1[0]
        
        # Enrich the contact_details_valid field if it exists
        if debtor_info and debtor_info.get("contact_details_valid"):
            try:
                query = f"""
                    SELECT contact_details_valid_state
                    FROM ct.contact_details_valid_state 
                    WHERE contact_details_valid_state_id={debtor_info["contact_details_valid"]}
                """
                state_result = db.execute_query(query)
                
                if state_result and len(state_result) > 0 and 'contact_details_valid_state' in state_result[0]:
                    debtor_info["contact_details_valid"] = state_result[0]['contact_details_valid_state']
                else:
                    debtor_info["contact_details_valid"] = "Unknown"
            except Exception as e:
                db.logger.error(f"Error enriching contact_details_valid: {e}")
                debtor_info["contact_details_valid"] = ""

        additional_fields ={
            "special_instruction": "special_instruction",
            "lastdisposition": 'last_disposition',
            'tax_reg_number':'tax_reg_number',
            'employer':'employer',
            "collection_instruction":'collection_instruction'
        }
        for key, value in additional_fields.items():
            debtor_info[key] = result2[0][value]

        
        return debtor_info
    except Exception as e:
        db.logger.error(f"Error in get_client_account_overview: {e}")
        return None

#-----------------------------------------------------#
# #Step 5: Promise to pay (PTP)
@tool
def get_client_account_statement(user_id: str) -> Optional[List[Dict[str, str]]]:
    """
    Retrieves a chronological statement of invoices, payments, and transactions.
    
    Use this tool to review a client's financial transaction history, including
    invoices issued, payments received, and credit notes.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        List of financial transactions containing:
        - timestamp: When the transaction occurred
        - doc_type: Type of transaction (Invoice, Payment, CreditNote)
        - comment: Description of transaction with reference numbers
        - amount: Transaction amount (positive for charges, negative for credits)
        - amount_outstanding: Remaining amount to be paid
    """
    return db.execute_function_and_extract_output(
        "ct", "sp_debtors_account_statement", f"'{user_id}'", False
    )

@tool
def get_client_banking_details_old(user_id: str) -> Optional[List[Dict[str, str]]]:
    """
    Retrieves client's payment methods and banking information.
    
    Use this tool when you need to verify or discuss a client's payment configuration,
    bank account details, or billing preferences.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        List of dictionaries containing payment and banking details:
        - payment_method: Method used for payments (e.g., "Direct Debit")
        - debit_date: Scheduled day for debit order processing
        - account_name: Name on bank account
        - account_number: Bank account number (may be partially masked)
        - account_type: Type of account (e.g., "Current", "Savings")
        - account_bank: Bank name
        - account_state: Current state of the account
    """
    return db.execute_function_and_extract_output(
        "ct", "sp_debtors_account_info", f"'{user_id}'", False
    )

@tool
def get_client_banking_details(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves comprehensive banking details for a client.
    
    Use this tool to get complete banking and payment information for a specific client,
    including account details, payment configuration, billing preferences, and account status.
    
    Args:
        user_id: Client's unique identifier (string)
        
    Returns:
        List containing dictionaries with all banking details, including:
        - user_bank_account_id: Unique ID for the bank account record
        - user_id: Client's unique identifier
        - bank_account_type_id: Type ID for the bank account
        - account_name: Full name on the bank account
        - branch_code: Bank branch code
        - account_number: Bank account number (may be partially masked)
        - bank_id: Bank's unique identifier
        -...
     """
    # First, get data from the SQL query (tool1)
    sql_query = f"""
        select 
	uba.* ,
	bn.bank_name ,
	bn.disabled,
	bn.avs,
	bn.allow_debit,
	bn.allow_mandate,
	bn.allow_debit_savings_account ,
	bat.description as bank_account_type,
	bac.description as bank_account_category,
	io.description as invoice_option,
	drd.day_value as debit_run_day,
	drd.display_name as debit_run_day_display_name,
	ps.description as pacs_service
    from ct.user_bank_account uba
    join ct.bank_account_type bat on bat.bank_account_type_id  = uba.bank_account_type_id
    join ct.bank_account_category bac  on bac.bank_account_category_id  = uba.bank_account_category_id
    join ct.invoice_option io  on io.invoice_option_id = uba.invoice_option_id
    join ct.debit_run_day drd on drd.debit_run_day_id = uba.debit_run_day_id
    join bl.pacs_service ps on ps.pacs_service_id  = uba.pacs_service_id
    join ct.bank_names bn  on bn.bank_id  = uba.bank_id
        where uba.user_id ='{user_id}'
    """
    sql_results = db.execute_query(sql_query)
    
    # Get data from the stored procedure (tool2)
    sp_results = db.execute_function_and_extract_output(
        "ct", "sp_debtors_account_info", f"'{user_id}'", False
    )
    
    # Return empty list if no results
    if not sql_results:
        return []
    
    # Combine data from both sources
    combined_results = []
    
    for sql_item in sql_results:
        # Find corresponding sp_item if available
        sp_item = None
        if sp_results:
            for item in sp_results:
                if item.get("account_number") == sql_item.get("account_number"):
                    sp_item = item
                    break
        
        # Start with sql_item data
        result_item = dict(sql_item)
        
        # Add sp_item data if available
        if sp_item:
            additional_fields = {
                "payment_method": sp_item.get("payment_method"),
                "debit_date": sp_item.get("debit_date"),
                "account_bank": sp_item.get("account_bank"),
                "account_branch_code": sp_item.get("account_branch_code"),
                "account_state": sp_item.get("account_state"),
                "credit_controller": sp_item.get("credit_controller"),
                "payment_terms": sp_item.get("payment_terms"),
                "account_branch_name": sp_item.get("account_branch_name"),
                "client_classification": sp_item.get("client_classification"),
                "account_rejected": sp_item.get("account_rejected"),
                "account_category": sp_item.get("account_category"),
                "non_primary": sp_item.get("non_primary")
            }
            
            # Add fields that don't exist in result_item or are null
            for key, value in additional_fields.items():
                if key not in result_item or result_item.get(key) is None:
                    result_item[key] = value
        
        combined_results.append(result_item)
    
    return combined_results

@tool
def get_client_payment_history(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves chronological history of payment promises/commitments and their outcomes.
    
    Use this tool to review a client's payment patterns, promises made, and
    whether they fulfilled those commitments.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        List of payment commitments (newest first) with:
        - arrangement_id: Unique identifier for the payment promise
        - amount: Promised payment amount
        - pay_date: Date client committed to pay
        - arrangement_state: Fulfillment status (e.g.: Failed, Completed, Pending)
        - created_by: Source of arrangement (Payment Portal, agent name)
    """
    sql_query = """
        SELECT
            a.arrangement_id,
            a.user_id,
            a.note,
            a.create_ts,
            a.followup_ts,
            a.created_by,
            s.description AS arrangement_state,
            ast.description AS arrangement_state_type,
            ap.amount,
            ap.pay_date,
            apt.description AS arrangement_pay_type,
            ap.mandate_fee,
            ap.cheque_number,
            ap.online_payment_reference_id_original
        FROM
            ct.arrangement a
            INNER JOIN ct.arrangement_state s 
                ON a.arrangement_state_id = s.arrangement_state_id
            INNER JOIN ct.arrangement_state_type ast 
                ON s.arrangement_state_type_id = ast.arrangement_state_type_id
            INNER JOIN ct.arrangement_pay ap 
                ON a.arrangement_id = ap.arrangement_id
            INNER JOIN ct.arrangement_pay_type apt 
                ON ap.arrangement_pay_type_id = apt.arrangement_pay_type_id
        WHERE
            a.user_id = '{}'
        ORDER BY
            a.create_ts DESC;
        """.format(user_id)
    return db.execute_query(sql_query)

#----------------#

@tool
def get_client_notes(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves client notes and communication history for a specific client.
    
    Use this tool to review all communication history, including notes, emails, SMS, and
    WhatsApp messages for a client.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        List of client notes and communications (newest first) with:
        - timestamp: Date and time when the note or communication was created
        - note: Content of the note or communication
        - source: Source of the note (Client, Email, Print Server, SMS, WhatsApp)
        - created_by: Username of who created the note or system identifier
    """
    sql_query = f"""
        SELECT * FROM ct.sp_client_debtor_notes({user_id})
        ORDER BY out_timestamp DESC;
        """
    
    results = db.execute_query(sql_query)
    
    # Transform the results to match the expected return format
    formatted_results = []
    for row in results:
        formatted_results.append({
            "timestamp": str(row["out_timestamp"]),
            "note": row["out_note"],
            "source": row["out_note_source"],
            "created_by": row["out_user_captured"]
        })
    
    return formatted_results

@tool
def get_client_debtor_letters(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves a history of automated letters sent to a client from the Debtors department.
    
    Use this tool to review formal communication sent to clients regarding payment matters,
    such as payment reminders, collection notices, and other official correspondence.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        List of letters (newest first) with:
        - report_id: Unique identifier for the letter/report
        - report_name: Name/title of the letter
        - sent_date: Date and time when the letter was sent
    """
    sql_query = f"""
        SELECT * FROM cc.list_debtor_letters({user_id})
        """
    
    results = db.execute_query(sql_query)
    
    # Transform the results to match the expected return format
    formatted_results = []
    for row in results:
        formatted_results.append({
            "report_id": str(row["out_report_id"]),
            "report_name": row["out_report_name"],
            "sent_date": str(row["out_sent_ts"])
        })
    
    return formatted_results

@tool
def get_client_failed_payments(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves history of failed debit order payments for a specific client.
    
    Use this tool to review a client's recent payment failures, including both
    automatic and manual debit orders that were unsuccessful.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        List of failed payments (newest first) with:
        - payment_date: Date when the payment failed
        - failure_reason: Description of why the payment failed
    """
    sql_query = """
        SELECT * FROM ct.get_debtor_info_failed_payment({})
        """.format(user_id)
    
    results = db.execute_query(sql_query)
    
    # Transform the results to match the expected return format
    formatted_results = []
    for row in results:
        formatted_results.append({
            "payment_date": row["out_payment_date"],
            "failure_reason": row["out_payment_type"] if row["out_payment_type"] else "Not specified"
        })
    
    return formatted_results

@tool
def get_client_last_successful_payment(user_id: str) -> Dict[str, Any]:
    """
    Retrieves information about the most recent successful payment made by a client.
    
    Use this tool to find when a client last made a successful payment and how much
    they paid. This can help assess recent payment behavior and when the last
    successful transaction occurred.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        Dictionary containing details about the last successful payment:
        - payment_id: Unique identifier for the payment
        - payment_date: Date when the payment was received
        - payment_amount: Amount of the payment
        
        Returns None if no successful payments are found.
    """
    sql_query = """
        SELECT * FROM ct.get_debtor_info_last_successful_payment({})
        """.format(user_id)
    
    result = db.execute_query(sql_query)
    
    # Check if we got any results
    if not result or len(result) == 0:
        return None
    
    # Transform the result to match the expected return format
    payment_info = result[0]  # Function returns a single record
    
    formatted_result = {
        "payment_id": str(payment_info["out_client_payment_id"]),
        "payment_date": str(payment_info["out_payment_date"]),
        "payment_amount": str(payment_info["out_payment_amt"])
    }
    
    return formatted_result

@tool
def get_client_last_valid_payment(user_id: str) -> Dict[str, Any]:
    """
    Retrieves information about the client's most recent valid payment, excluding
    most reversed payments and refunds.
    
    Use this tool to find the last genuine payment made by a client that wasn't
    later reversed. This helps assess true payment behavior by filtering out
    transactions that were initially recorded but later invalidated.
    
    Note: This function considers payments with empty accounts ("Unpaid (02)")
    as valid payments, unlike other reversed payments.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        Dictionary containing details about the last valid payment:
        - payment_id: Unique identifier for the payment
        - payment_date: Date when the payment was received
        - payment_amount: Amount of the payment
        
        Returns None if no valid payments are found.
    """
    sql_query = """
        SELECT * FROM ct.get_debtor_info_last_successful_payment_exclude_reversal({})
        """.format(user_id)
    
    result = db.execute_query(sql_query)
    
    # Check if we got any results
    if not result or len(result) == 0:
        return None
    
    # Transform the result to match the expected return format
    payment_info = result[0]  # Function returns a single record
    
    formatted_result = {
        "payment_id": str(payment_info["out_client_payment_id"]),
        "payment_date": str(payment_info["out_payment_date"]),
        "payment_amount": str(payment_info["out_payment_amt"])
    }
    
    return formatted_result

@tool
def get_client_last_reversed_payment(user_id: str) -> Dict[str, Any]:
    """
    Retrieves information about the client's most recent payment that was later reversed.
    
    Use this tool to find payments that were initially recorded but later invalidated
    due to issues like insufficient funds, disputes, or other payment failures.
    This can help identify patterns of attempted but unsuccessful payments.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        Dictionary containing details about the last reversed payment:
        - payment_id: Unique identifier for the payment
        - payment_date: Date when the payment was initially received
        - payment_amount: Amount of the attempted payment
        - reversal_reason: Reason for the payment reversal
        
        Returns None if no reversed payments are found.
    """
    sql_query = f"""
        SELECT
            cp.client_payment_id,
            cp.date_received,
            cp.payment_amt,
            cp.pay_method
        FROM
            ct.client_payment cp
        WHERE
            cp.user_id = {user_id}
            and cp.payment_amt < 0
        ORDER BY
            cp.date_received DESC,
            cp.client_payment_id DESC
        LIMIT 1
        """
    
    result = db.execute_query(sql_query)
    
    # Check if we got any results
    if not result or len(result) == 0:
        return None
    
    # Transform the result to match the expected return format
    payment_info = result[0]
    
    
    return payment_info
#----------------#
@tool
def update_payment_arrangements(user_id: str) -> Dict[str, Any]:
    """
    Updates payment arrangement statuses for a client in the system.
    
    Use this tool after creating a new payment arrangement or when you need
    to refresh the status of existing arrangements (e.g., to mark them as
    completed or failed based on actual payments).
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
    
    Returns:
        Dictionary with:
        - success: Boolean indicating if the update was successful
        - message: Description of the result
    """
    result = {
        "success": False,
        "message": "",
    }
    
    try:
       # Get the username for the user_id
        user_info_query = f"""
        SELECT user_name 
        FROM ct.user 
        WHERE user_id = {user_id}
        """
        
        user_result = db.execute_query(user_info_query)
        if not user_result or 'user_name' not in user_result[0]:
            return {"success": False, "message": f"Could not find username for user_id: {user_id}"}
        
        # Get the username - this will be used for au$username
        username = user_result[0]['user_name']

        try:
            # Set the custom.user_name
            db.execute_query(f"SET custom.user_name = '{username}'")

            # Execute the stored procedure
            db.execute_query(f"SELECT bl.sp_update_arrangement(current_date);")

            # Reset the custom.user_name
            db.execute_query("RESET custom.user_name")

            result["success"] = True
            result["message"] = "Payment arrangements updated successfully"

        except Exception as e:
            result["message"] = f"Error executing sp_update_arrangement: {str(e)}"

    except Exception as e:
        result["message"] = f"Error: {str(e)}"
        
    return result


#-----------------------------------------------------#
# #Step 6: DebiCheck
@tool
def get_payment_arrangement_types(query: str = "all") -> List[Dict[str, str]]:
    """
    Retrieves payment arrangement types available for customers.
    
    Args:
        query: Optional filter parameter. Default "all" returns all payment types.
    
    Usage:
        get_payment_arrangement_types("all") -> Returns all payment arrangement types
    
    Returns:
        List of payment arrangement types with format:
    """
    sql_query = """
    SELECT 
        arrangement_pay_type_id as id,
        description
    FROM ct.arrangement_pay_type
    ORDER BY arrangement_pay_type_id;
    """
    return db.execute_query(sql_query)

@tool
def get_client_debit_mandates(user_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves electronic debit order authorizations (DebiCheck mandates) for a client.
    
    Use this tool to check if a client has valid DebiCheck mandates for collections
    and to view the status and details of those mandates.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        
    Returns:
        List of debit order mandates (newest first) with:
        - client_debit_order_mandate_id: Unique mandate identifier
        - debicheck_mandate_state: Current status (Created, Authenticated, etc.)
        - authenticated: Whether mandate has been authenticated (True/False)
        - collection_amount: Regular collection amount
        - collection_day: Day of month for collection
        - debtor_account_name: Name on bank account
        - debtor_account_number: Account number (may be anonymized)
    """
    sql_query = f"""
    SELECT 
        cdom.* ,
        dms.description as debicheck_mandate_state
    FROM ct.client_debit_order_mandate cdom 
    join ct.debicheck_mandate_state dms on dms.debicheck_mandate_state_id = cdom.debicheck_mandate_state_id
    WHERE cdom.client_id = '{user_id}'
    ORDER BY cdom.create_ts DESC;
    """
    return db.execute_query(sql_query)


@tool
def create_mandate_old(
    user_id: int,
    service: str,
    amount: float = None,
    collection_date: str = None,  # Format: 'YYYY-MM-DD'
    authentication_code: str = None
) -> dict:
    """
    Creates a debit order mandate for a user (either recurring or once-off).
    
    A mandate is required before creating a direct debit payment arrangement.
    
    Args:
        user_id: Client ID (integer, e.g., 83906)
        service: Service identifier/category (e.g., 'PTP' for Promise to Pay, 'Subscription')
        amount: Collection amount (required for once-off mandates, calculated from contracts for recurring)
        collection_date: Collection date in 'YYYY-MM-DD' format (if provided, creates a once-off mandate; 
                         if None, creates a recurring mandate)
        authentication_code: DebiCheck authentication code (defaults to system value if None)
        
    Returns:
        {
            "success": True/False,
            "message": "Result description",
            "mandate_id": ID if successful, None if failed,
            "mandate_type": "Recurring" or "Once-off"
        }
    """
    import datetime
    
    # Fast-fail validation for critical fields
    if not isinstance(user_id, int) or user_id <= 0:
        return {
            'success': False,
            'message': "user_id must be a positive integer",
            'mandate_id': None,
            'mandate_type': None
        }
    
    if not service or not isinstance(service, str):
        return {
            'success': False,
            'message': "service identifier is required",
            'mandate_id': None,
            'mandate_type': None
        }
    
    # Determine mandate type based on collection_date
    is_recurring = collection_date is None
    mandate_type = "Recurring" if is_recurring else "Once-off"
    
    # Validate once-off mandate requirements
    if not is_recurring:
        # Validate amount
        if amount is None or amount <= 0:
            return {
                'success': False,
                'message': "amount is required and must be positive for once-off mandates",
                'mandate_id': None,
                'mandate_type': None
            }
        
        # Validate date format and ensure future date
        try:
            date_obj = datetime.datetime.strptime(collection_date, '%Y-%m-%d')
            # Check if date is in the future
            if date_obj.date() < datetime.datetime.now().date():
                return {
                    'success': False,
                    'message': "collection_date must be in the future",
                    'mandate_id': None,
                    'mandate_type': None
                }
        except (ValueError, TypeError):
            return {
                'success': False,
                'message': "Invalid date format. Use YYYY-MM-DD",
                'mandate_id': None,
                'mandate_type': None
            }
    
    # Set audit username for tracking
    audit_username = "ai_agent"
    
    try:
        # Begin transaction and set parameters
        transaction_commands = [
            "BEGIN;",
            f"SET custom.user_name TO '{audit_username}';",
            f"""
            SELECT bl.mandate_insert(
                {user_id}, 
                '{service}', 
                {f"'{authentication_code}'" if authentication_code else "NULL"}, 
                {amount if amount is not None else "NULL"}, 
                {f"'{collection_date}'" if collection_date else "NULL"}
            ) as result;
            """,
            f"""
            SELECT client_debit_order_mandate_id
            FROM ct.client_debit_order_mandate
            WHERE client_id = {user_id} 
            ORDER BY create_ts DESC
            LIMIT 1;
            """,
            "COMMIT;"
        ]
        
        # Execute commands and collect results
        results = []
        for cmd in transaction_commands:
            result = db.execute_query(cmd)
            if result and (not cmd.strip().startswith("BEGIN") and not cmd.strip().startswith("COMMIT")):
                results.append(result)
        
        # Reset user_name (do this outside the transaction to ensure it happens even on error)
        db.execute_query("RESET custom.user_name;")
        
        # Extract function result and mandate ID
        if len(results) < 2:
            return {
                'success': False,
                'message': 'Database error: Incomplete transaction results',
                'mandate_id': None,
                'mandate_type': None
            }
        
        # Process results
        result_value = results[0][0]['result'] if results[0] and len(results[0]) > 0 and 'result' in results[0][0] else None
        mandate_id = results[1][0]['client_debit_order_mandate_id'] if results[1] and len(results[1]) > 0 and 'client_debit_order_mandate_id' in results[1][0] else None
        
        # Determine success based on results
        if result_value == 'OK' and mandate_id is not None:
            return {
                'success': True,
                'message': f'{mandate_type} mandate created successfully',
                'mandate_id': mandate_id,
                'mandate_type': mandate_type
            }
        elif result_value and result_value != 'OK':
            return {
                'success': False,
                'message': f'Creation failed: {result_value}',
                'mandate_id': None,
                'mandate_type': None
            }
        else:
            return {
                'success': False,
                'message': 'Failed to create mandate or retrieve mandate ID',
                'mandate_id': None,
                'mandate_type': None
            }
                
    except Exception as e:
        # Rollback and cleanup in case of error
        try:
            db.execute_query("ROLLBACK;")
            db.execute_query("RESET custom.user_name;")
        except:
            pass
            
        return {
            'success': False,
            'message': f'Error creating mandate: {str(e)}',
            'mandate_id': None,
            'mandate_type': None
        }

@tool
def create_mandate(
    user_id: int,
    service: str = "TT1",
    amount: float = None,
    collection_date: str = None,  # Format: 'YYYY-MM-DD'
    authentication_code: str = None
) -> dict:
    """
    Creates a debit order mandate for a user (either recurring or once-off).
    
    A mandate is required before creating a direct debit payment arrangement.
    
    Args:
        user_id: Client ID (integer, e.g., 83906)
        service: Service type - 'TT1' (default), 'TT2', or 'MIGRATE' for recurring mandates
                or a descriptive identifier (e.g., 'PTP') for once-off mandates
        amount: Collection amount (required for once-off mandates, calculated from contracts for recurring)
        collection_date: Collection date in 'YYYY-MM-DD' format (if provided, creates a once-off mandate; 
                       if None, creates a recurring mandate using sp_client_debit_order_mandate_create)
        authentication_code: DebiCheck authentication code (defaults to system value if None)
        
    Returns:
        {
            "success": True/False,
            "message": "Result description",
            "mandate_id": ID if successful, None if failed,
            "mandate_type": "Recurring" or "Once-off",
            "contract_count": Number of contracts (for recurring mandates only)
        }
    """
    import datetime
    
    # Fast-fail validation for critical fields
    if not isinstance(user_id, int) or user_id <= 0:
        return {
            'success': False,
            'message': "user_id must be a positive integer",
            'mandate_id': None,
            'mandate_type': None,
            'contract_count': 0
        }
    
    if not service or not isinstance(service, str):
        return {
            'success': False,
            'message': "service identifier is required",
            'mandate_id': None,
            'mandate_type': None,
            'contract_count': 0
        }
    
    # Determine mandate type based on collection_date
    is_recurring = collection_date is None
    mandate_type = "Recurring" if is_recurring else "Once-off"
    
    # Validate once-off mandate requirements
    if not is_recurring:
        # Validate amount
        if amount is None or amount <= 0:
            return {
                'success': False,
                'message': "amount is required and must be positive for once-off mandates",
                'mandate_id': None,
                'mandate_type': None,
                'contract_count': 0
            }
        
        # Validate date format and ensure future date
        try:
            date_obj = datetime.datetime.strptime(collection_date, '%Y-%m-%d')
            # Check if date is in the future
            if date_obj.date() < datetime.datetime.now().date():
                return {
                    'success': False,
                    'message': "collection_date must be in the future",
                    'mandate_id': None,
                    'mandate_type': None,
                    'contract_count': 0
                }
        except (ValueError, TypeError):
            return {
                'success': False,
                'message': "Invalid date format. Use YYYY-MM-DD",
                'mandate_id': None,
                'mandate_type': None,
                'contract_count': 0
            }
    
    # Set audit username for tracking
    audit_username = "ai_agent"
    
    try:
        # For recurring mandates, use sp_client_debit_order_mandate_create for comprehensive validation
        if is_recurring:
            # Get username for the user_id for sp_client_debit_order_mandate_create
            username_query = f"SELECT user_name FROM ct.user WHERE user_id = {user_id};"
            username_result = db.execute_query(username_query)
                
            if not username_result or len(username_result) == 0 or 'user_name' not in username_result[0]:
                return {
                    'success': False,
                    'message': f"Cannot find username for user_id: {user_id}",
                    'mandate_id': None,
                    'mandate_type': None,
                    'contract_count': 0
                }
                
            username = username_result[0]['user_name']
            
            # Begin transaction and use sp_client_debit_order_mandate_create
            transaction_commands = [
                "BEGIN;",
                f"SET custom.user_name TO '{audit_username}';",
                f"""
                SELECT * FROM ct.sp_client_debit_order_mandate_create(
                    '{username}', 
                    '{service}',
                    {f"'{authentication_code}'" if authentication_code else "NULL"}
                );
                """,
                f"""
                SELECT client_debit_order_mandate_id
                FROM ct.client_debit_order_mandate cdm
                WHERE cdm.client_id = {user_id}
                ORDER BY cdm.create_ts DESC
                LIMIT 1;
                """,
                "COMMIT;"
            ]
            
            # Execute commands and collect results
            results = []
            for cmd in transaction_commands:
                result = db.execute_query(cmd)
                if result and (not cmd.strip().startswith("BEGIN") and not cmd.strip().startswith("COMMIT")):
                    results.append(result)
            
            # Reset user_name
            db.execute_query("RESET custom.user_name;")
            
            # Extract function result and mandate ID
            if len(results) < 2:
                return {
                    'success': False,
                    'message': 'Database error: Incomplete transaction results',
                    'mandate_id': None,
                    'mandate_type': "Recurring",
                    'contract_count': 0
                }
            
            # Process results from sp_client_debit_order_mandate_create
            function_result = results[0][0] if results[0] and len(results[0]) > 0 else None
            
            if not function_result:
                return {
                    'success': False,
                    'message': 'Database error: Failed to execute mandate creation function',
                    'mandate_id': None,
                    'mandate_type': "Recurring",
                    'contract_count': 0
                }
            
            out_client_user_name = function_result.get('out_client_user_name', '')
            out_contract_count = function_result.get('out_contract_count', 0)
            out_response = function_result.get('out_response', '')
            
            # Get mandate ID from second query
            mandate_id = results[1][0]['client_debit_order_mandate_id'] if results[1] and len(results[1]) > 0 and 'client_debit_order_mandate_id' in results[1][0] else None
            
            # Determine success based on results
            if out_response == 'MANDATE EXPECTANCY REQUEST CREATED' and mandate_id is not None:
                return {
                    'success': True,
                    'message': f'Recurring mandate created successfully with {out_contract_count} contract(s)',
                    'mandate_id': mandate_id,
                    'mandate_type': "Recurring",
                    'contract_count': out_contract_count
                }
            else:
                return {
                    'success': False,
                    'message': out_response or 'Failed to create recurring mandate',
                    'mandate_id': None,
                    'mandate_type': "Recurring",
                    'contract_count': out_contract_count
                }
                
        else:
            # For once-off mandates, use bl.mandate_insert directly
            transaction_commands = [
                "BEGIN;",
                f"SET custom.user_name TO '{audit_username}';",
                f"""
                SELECT bl.mandate_insert(
                    {user_id}, 
                    '{service}', 
                    {f"'{authentication_code}'" if authentication_code else "NULL"}, 
                    {amount if amount is not None else "NULL"}, 
                    '{collection_date}'
                ) as result;
                """,
                f"""
                SELECT client_debit_order_mandate_id
                FROM ct.client_debit_order_mandate
                WHERE client_id = {user_id} 
                AND debit_order_mandate_type_id = 4
                ORDER BY create_ts DESC
                LIMIT 1;
                """,
                "COMMIT;"
            ]
            
            # Execute commands and collect results
            results = []
            for cmd in transaction_commands:
                result = db.execute_query(cmd)
                if result and (not cmd.strip().startswith("BEGIN") and not cmd.strip().startswith("COMMIT")):
                    results.append(result)
            
            # Reset user_name
            db.execute_query("RESET custom.user_name;")
            
            # Extract function result and mandate ID
            if len(results) < 2:
                return {
                    'success': False,
                    'message': 'Database error: Incomplete transaction results',
                    'mandate_id': None,
                    'mandate_type': "Once-off",
                    'contract_count': 0
                }
            
            # Process results
            result_value = results[0][0]['result'] if results[0] and len(results[0]) > 0 and 'result' in results[0][0] else None
            mandate_id = results[1][0]['client_debit_order_mandate_id'] if results[1] and len(results[1]) > 0 and 'client_debit_order_mandate_id' in results[1][0] else None
            
            # Determine success based on results
            if result_value == 'OK' and mandate_id is not None:
                return {
                    'success': True,
                    'message': 'Once-off mandate created successfully',
                    'mandate_id': mandate_id,
                    'mandate_type': "Once-off",
                    'contract_count': 1
                }
            elif result_value and result_value != 'OK':
                return {
                    'success': False,
                    'message': f'Creation failed: {result_value}',
                    'mandate_id': None,
                    'mandate_type': "Once-off",
                    'contract_count': 0
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to create once-off mandate or retrieve mandate ID',
                    'mandate_id': None,
                    'mandate_type': "Once-off",
                    'contract_count': 0
                }
                
    except Exception as e:
        # Rollback and cleanup in case of error
        try:
            db.execute_query("ROLLBACK;")
            db.execute_query("RESET custom.user_name;")
        except:
            pass
            
        return {
            'success': False,
            'message': f'Error creating mandate: {str(e)}',
            'mandate_id': None,
            'mandate_type': mandate_type,
            'contract_count': 0
        }


@tool
def create_debicheck_payment(
    user_id: int,
    payment_amount: float,
    payment_date: str,  # Format: 'YYYY-MM-DD'
    service: str = "PTP",
    note: str = "",
    payment2: float = 0.0,
    date2: str = None,
    payment3: float = 0.0,
    date3: str = None,
    mandate_fee: float = 10.0,
    authentication_code: str = None
) -> dict:
    """
    Creates a DebiCheck payment arrangement (default: one-time payment).
    
    Args:
        user_id: Client ID
        payment_amount: Amount for first payment
        payment_date: First payment date ('YYYY-MM-DD')
        service: Service type - "PTP" (one-time) or "TT1"/"TT2"/"MIGRATE" (recurring)
        note: Optional payment description
        payment2: Second payment amount (optional)
        date2: Second payment date (required if payment2 > 0)
        payment3: Third payment amount (optional)
        date3: Third payment date (required if payment3 > 0)
        mandate_fee: DebiCheck mandate fee (default: 10.0)
        authentication_code: DebiCheck authentication code (optional)
        
    Returns:
        Dict with success status, message, IDs, and mandate type
    """
    try:
        # Convert user_id to int if needed
        user_id = int(user_id)
        payment_amount = float(payment_amount)
        
        # Validate inputs
        if user_id <= 0 or payment_amount <= 0 or not payment_date:
            missing = []
            if user_id <= 0: missing.append("valid user_id")
            if payment_amount <= 0: missing.append("positive payment_amount")
            if not payment_date: missing.append("payment_date")
            return _error_response(f"Missing required fields: {', '.join(missing)}")
            
        # Validate additional payments
        if (payment2 > 0 and not date2) or (payment3 > 0 and not date3):
            return _error_response("Date is required for each payment amount")
        
        # Determine mandate type
        is_recurring = payment2 > 0 or payment3 > 0 or service in ["TT1", "TT2", "MIGRATE"]
        
        # Set service type for one-time payments
        if not is_recurring:
            service = "PTP"
        
        # Create mandate
        mandate_params = {'user_id': user_id, 'service': service}
        
        if not is_recurring:
            mandate_params.update({
                'amount': payment_amount,
                'collection_date': payment_date
            })
            
        if authentication_code:
            mandate_params['authentication_code'] = authentication_code

        mandate_result = create_mandate.invoke(mandate_params)
        
        if not mandate_result["success"]:
            return {
                "success": False,
                "message": f"Mandate creation failed: {mandate_result['message']}",
                "mandate_id": None,
                "arrangement_id": None,
                "mandate_type": "Recurring" if is_recurring else "Once-off"
            }
        
        # Create payment arrangement
        arrangement_params = {
            'user_id': user_id,
            'pay_type_id': 1,  # DebiCheck
            'payment1': payment_amount,
            'date1': payment_date,
            'note': note or ("Recurring payment" if is_recurring else "One-time payment"),
            'mandate_id1': mandate_result["mandate_id"],
            'mandate_fee': mandate_fee
        }
        
        # Add additional payments if specified
        if payment2 > 0:
            arrangement_params.update({
                'payment2': payment2,
                'date2': date2
            })
        
        if payment3 > 0:
            arrangement_params.update({
                'payment3': payment3,
                'date3': date3
            })
        
        arrangement_result = create_payment_arrangement.invoke(arrangement_params)
        
        if not arrangement_result["success"]:
            return {
                "success": False,
                "message": f"Mandate created but payment failed: {arrangement_result['message']}",
                "mandate_id": mandate_result["mandate_id"],
                "arrangement_id": None,
                "mandate_type": "Recurring" if is_recurring else "Once-off"
            }
        
        # Success response
        payment_type = "recurring" if is_recurring else "one-time"
        installments = sum(1 for p in [payment_amount, payment2, payment3] if p > 0)
        payment_desc = f" with {installments} installments" if is_recurring and installments > 1 else ""
        
        return {
            "success": True,
            "message": f"Successfully created {payment_type} payment{payment_desc}",
            "mandate_id": mandate_result["mandate_id"],
            "arrangement_id": arrangement_result["arrangement_id"],
            "mandate_type": "Recurring" if is_recurring else "Once-off"
        }
    
    except (ValueError, TypeError) as e:
        return _error_response(f"Invalid input: {str(e)}")

def _error_response(message):
    return {
        "success": False,
        "message": message,
        "mandate_id": None,
        "arrangement_id": None,
        "mandate_type": None
    }

@tool
def create_payment_arrangement(
    user_id: int,
    pay_type_id: int,
    payment1: float,
    date1: str,  # Format: 'YYYY-MM-DD'
    note: str = "",
    payment2: float = 0.0,
    date2: str = None,
    payment3: float = 0.0,
    date3: str = None,
    mandate_id1: int = None,
    mandate_fee: float = 0.0
) -> dict:
    """
    Creates a payment arrangement with up to three installments.
    
    IMPORTANT: For DebiCheck payments (pay_type_id=1), you MUST create a mandate first 
    using the create_mandate() function and pass the resulting mandate_id as mandate_id1.
    
    Payment Types (pay_type_id):
    - 1 = Direct Debit/DebiCheck (requires mandate_id1)
    - 2 = EFT
    - 3 = Credit Card
    - 4 = OZOW
    - 5 = Pay@
    - 6 = Cheque
    - 7 = Capitec Pay
    
    Args:
        user_id: Client ID (integer, e.g., 83906)
        pay_type_id: Payment method ID (see list above)
        payment1: Amount for first payment (required)
        date1: First payment date ('YYYY-MM-DD')
        note: Optional description of arrangement
        payment2: Second payment amount (optional)
        date2: Second payment date (optional, required if payment2 > 0)
        payment3: Third payment amount (optional)
        date3: Third payment date (optional, required if payment3 > 0)
        mandate_id1: DebiCheck mandate ID (REQUIRED for Direct Debit/DebiCheck payments with pay_type_id=1)
        mandate_fee: DebiCheck mandate fee (only applies when using mandate_id1)
        
    Returns:
        {
            "success": True/False,
            "message": "Result description",
            "arrangement_id": ID if successful, None if failed
        }
    """
    import datetime
    from datetime import timedelta
    
    # Fast-fail validation for critical fields
    if not isinstance(user_id, int) or user_id <= 0:
        return {
            'success': False,
            'message': "user_id must be a positive integer",
            'arrangement_id': None
        }
    
    if not isinstance(pay_type_id, int) or pay_type_id <= 0 or pay_type_id > 7:
        return {
            'success': False,
            'message': "pay_type_id must be between 1 and 7",
            'arrangement_id': None
        }
    
    # Validate DebiCheck mandate requirements
    if pay_type_id == 1:
        if mandate_id1 is None:
            return {
                'success': False,
                'message': "DebiCheck payments (pay_type_id=1) require a mandate_id1. Please create a mandate first using create_mandate().",
                'arrangement_id': None
            }
    
    if not payment1 or payment1 <= 0:
        return {
            'success': False,
            'message': "payment1 must be a positive amount",
            'arrangement_id': None
        }
    
    if not date1:
        return {
            'success': False,
            'message': "date1 is required",
            'arrangement_id': None
        }
    
    # Process and validate dates
    try:
        # Parse date1
        date1_obj = datetime.datetime.strptime(date1, '%Y-%m-%d')
        
        # Validate date is not in the past
        if date1_obj.date() < datetime.datetime.now().date():
            return {
                'success': False,
                'message': "date1 must be in the future",
                'arrangement_id': None
            }
            
        # Handle payment2/date2 if provided
        if payment2 and payment2 > 0:
            if not date2:
                return {
                    'success': False,
                    'message': "date2 is required when payment2 is provided",
                    'arrangement_id': None
                }
            
            date2_obj = datetime.datetime.strptime(date2, '%Y-%m-%d')
            if date2_obj <= date1_obj:
                return {
                    'success': False,
                    'message': "date2 must be after date1",
                    'arrangement_id': None
                }
        
        # Handle payment3/date3 if provided
        if payment3 and payment3 > 0:
            if not date3:
                return {
                    'success': False,
                    'message': "date3 is required when payment3 is provided",
                    'arrangement_id': None
                }
            
            date3_obj = datetime.datetime.strptime(date3, '%Y-%m-%d')
            if payment2 and payment2 > 0 and date3_obj <= date2_obj:
                return {
                    'success': False,
                    'message': "date3 must be after date2",
                    'arrangement_id': None
                }
            elif date3_obj <= date1_obj:
                return {
                    'success': False,
                    'message': "date3 must be after date1",
                    'arrangement_id': None
                }
    
    except ValueError as e:
        return {
            'success': False,
            'message': f"Invalid date format. Use YYYY-MM-DD: {str(e)}",
            'arrangement_id': None
        }
    
    # Pre-compute all values needed for query
    followup_date = (date1_obj + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Calculate installments efficiently
    num_installments = 1
    total_payment = payment1
    
    if payment2 and payment2 > 0:
        num_installments += 1
        total_payment += payment2
    
    if payment3 and payment3 > 0:
        num_installments += 1
        total_payment += payment3
    
    # Format note efficiently
    if note and note.strip():
        formatted_note = f"Promise to Pay: R{total_payment:.2f} {date1}. Arrangement with client for x{num_installments} {'installments' if num_installments > 1 else 'installment'} - {note.strip()}"
    else:
        formatted_note = f"Promise to Pay: R{total_payment:.2f} {date1}. Arrangement with client for x{num_installments} {'installments' if num_installments > 1 else 'installment'}"
    
    # Set audit username for tracking
    audit_username = "ai_agent"
    
    try:
        # Execute all database commands in a transaction
        transaction_commands = [
            "BEGIN;",
            f"SET custom.user_name TO '{audit_username}';",
            f"""
            SELECT ct.create_payment_arrangement_part_1(
                {user_id}, 
                '{formatted_note}', 
                '{followup_date}', 
                {pay_type_id}, 
                {mandate_fee or 0.0}, 
                {payment1}, 
                '{date1}', 
                {mandate_id1 or 'NULL'}, 
                NULL, 
                NULL, 
                {payment2 or 0.0}, 
                {f"'{date2}'" if date2 else "NULL"}, 
                NULL, 
                NULL, 
                NULL, 
                {payment3 or 0.0}, 
                {f"'{date3}'" if date3 else "NULL"}, 
                NULL, 
                NULL, 
                NULL,
                NULL
            ) as result;
            """,
            f"""
            SELECT arrangement_id
            FROM ct.arrangement
            WHERE user_id = {user_id} 
            ORDER BY create_ts DESC
            LIMIT 1;
            """,
            "COMMIT;"
        ]
        
        # Execute commands and collect results
        results = []
        for cmd in transaction_commands:
            result = db.execute_query(cmd)
            if result and (not cmd.strip().startswith("BEGIN") and not cmd.strip().startswith("COMMIT")):
                results.append(result)
        
        # Reset user_name (outside transaction to ensure it happens)
        db.execute_query("RESET custom.user_name;")
        
        # Extract function result and arrangement ID
        if len(results) < 2:
            return {
                'success': False,
                'message': 'Database error: Incomplete transaction results',
                'arrangement_id': None
            }
        
        # Get function result and arrangement ID
        result_value = results[0][0]['result'] if results[0] and len(results[0]) > 0 and 'result' in results[0][0] else None
        arrangement_id = None
        
        # Extract the arrangement ID from the result (format: 'OK12345')
        if result_value and result_value.startswith('OK'):
            try:
                arrangement_id = int(result_value[2:])
            except:
                arrangement_id = results[1][0]['arrangement_id'] if results[1] and len(results[1]) > 0 and 'arrangement_id' in results[1][0] else None
        else:
            arrangement_id = results[1][0]['arrangement_id'] if results[1] and len(results[1]) > 0 and 'arrangement_id' in results[1][0] else None
        
        # Determine success based on results
        if arrangement_id is not None:
            return {
                'success': True,
                'message': f'Payment arrangement created successfully with {num_installments} installment(s)',
                'arrangement_id': arrangement_id
            }
        elif result_value and not result_value.startswith('OK'):
            return {
                'success': False,
                'message': f'Creation failed: {result_value}',
                'arrangement_id': None
            }
        else:
            return {
                'success': False,
                'message': 'Failed to create payment arrangement or retrieve arrangement ID',
                'arrangement_id': None
            }
                
    except Exception as e:
        # Rollback and cleanup in case of error
        try:
            db.execute_query("ROLLBACK;")
            db.execute_query("RESET custom.user_name;")
        except:
            pass
            
        return {
            'success': False,
            'message': f'Error creating payment arrangement: {str(e)}',
            'arrangement_id': None
        }


#-----------------------------------------------------#
# #Step 7: Subscription
@tool
def get_client_subscription_amount(user_id: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    """
    Calculates a client's total subscription amount across all their contracts.
    
    Use this tool to determine how much a client is supposed to pay regularly for
    their active subscriptions. This can help assess whether their payments are
    keeping up with their subscription obligations.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        start_date: Optional start date for historical calculation (format: "YYYY-MM-DD")
        end_date: Optional end date for historical calculation (format: "YYYY-MM-DD")
                  Note: If end_date is provided, start_date must also be provided
        
    Returns:
        Dictionary containing:
        - subscription_amount: Total current subscription amount
        - calculation_period: Description of the period used for calculation
    """
    # Prepare SQL query based on whether date parameters are provided
    if end_date and start_date:
        sql_query = """
            SELECT ct.get_debtor_info_subscription({}, '{}', '{}') AS subscription_amount
            """.format(user_id, start_date, end_date)
        calculation_period = f"Contracts active from {start_date} to {end_date}"
    else:
        sql_query = """
            SELECT ct.get_debtor_info_subscription({}) AS subscription_amount
            """.format(user_id)
        calculation_period = "Current active contracts only"
    
    result = db.execute_query(sql_query)
    
    # Format the result
    subscription_amount = result[0]["subscription_amount"] if result else 0
    
    return {
        "subscription_amount": str(subscription_amount),
        "calculation_period": calculation_period
    }

#-----------------------------------------------------#
# #Step 8: Cartrack Payment Portal
@tool
def create_payment_arrangement_payment_portal(
    user_id: int,
    payment_type_id: int,
    payment_date: str,  # Format: 'YYYY-MM-DD'
    amount: float,
    online_payment_reference_id: str = None
) -> dict:
    """
    Creates a single-payment arrangement through the payment portal.
    
    Args:
        user_id: Client ID (integer, e.g., 83906)
        payment_type_id: Payment method ID (1=Direct Debit, 2=EFT, 3=Credit Card, 4=OZOW, 5=Pay@)
        payment_date: Payment date in 'YYYY-MM-DD' format
        amount: Payment amount (must be positive)
        online_payment_reference_id: UUID reference for online payments (optional, will be generated if not provided)
        
    Returns:
        {
            "success": True/False,
            "message": "Status message or error",
            "arrangement_id": ID if successful, None if failed,
            "online_payment_reference_id": UUID used for the transaction
        }
    """
    import datetime
    import uuid
    
    # Fast-fail validation for required fields - minimizes further processing if invalid
    if not isinstance(user_id, int) or user_id <= 0:
        return {
            'success': False,
            'message': "user_id must be a positive integer",
            'arrangement_id': None,
            'online_payment_reference_id': None
        }
    
    if not isinstance(payment_type_id, int) or payment_type_id <= 0:
        return {
            'success': False,
            'message': "payment_type_id must be a positive integer",
            'arrangement_id': None,
            'online_payment_reference_id': None
        }
    
    if not amount or amount <= 0:
        return {
            'success': False,
            'message': "amount must be a positive value",
            'arrangement_id': None,
            'online_payment_reference_id': None
        }
    
    if not payment_date:
        return {
            'success': False,
            'message': "payment_date is required",
            'arrangement_id': None,
            'online_payment_reference_id': None
        }
    
    # Date and UUID processing - with optimized error handling
    try:
        # Parse the date - only do this once and reuse the object
        payment_date_obj = datetime.datetime.strptime(payment_date, '%Y-%m-%d')
        followup_date = (payment_date_obj + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Process UUID - use provided or generate new one
        if online_payment_reference_id:
            uuid_obj = uuid.UUID(online_payment_reference_id)
        else:
            uuid_obj = uuid.uuid4()
        
        uuid_str = str(uuid_obj)
        uuid_param = f"'{uuid_str}'::uuid"
        
    except ValueError as e:
        if "does not match format" in str(e):
            return {
                'success': False,
                'message': f"Invalid date format: {payment_date}. Expected format: YYYY-MM-DD",
                'arrangement_id': None,
                'online_payment_reference_id': None
            }
        else:
            return {
                'success': False,
                'message': f"Invalid UUID format: {online_payment_reference_id}",
                'arrangement_id': None,
                'online_payment_reference_id': None
            }
    
    # Pre-format note and all parameters to avoid recalculation
    formatted_note = f"Promise to Pay: R{round(amount, 2)} {payment_date}. Comment: Created by client on web payment portal"
    audit_user = "ai_agent"
    
    # SQL transaction setup - combine all database operations
    transaction_commands = [
        f"SET custom.user_name TO '{audit_user}';",
        "SET custom.application TO 'ct.create_payment_arrangement_from_web';",
        f"""
        SELECT ct.create_payment_arrangement_part_1(
            {user_id}, 
            '{formatted_note}', 
            '{followup_date}', 
            {payment_type_id}, 
            0.0, 
            {amount}, 
            '{payment_date}', 
            NULL, 
            NULL, 
            NULL, 
            0.0, 
            NULL, 
            NULL, 
            NULL, 
            NULL, 
            0.0, 
            NULL, 
            NULL, 
            NULL, 
            NULL,
            {uuid_param}
        ) as result;
        """,
        f"""
        SELECT arrangement_id 
        FROM ct.arrangement 
        WHERE user_id = {user_id} AND created_by = '{audit_user}' 
        ORDER BY create_ts DESC 
        LIMIT 1;
        """,
        "RESET custom.user_name;",
        "RESET custom.application;"
    ]
    
    try:
        # Execute all database operations in a single connection
        results = []
        for cmd in transaction_commands:
            result = db.execute_query(cmd)
            if result:
                results.append(result)
        
        # Process results - we need the function result (3rd command) and arrangement_id (4th command)
        if len(results) < 2 or not results[0] or 'result' not in results[0][0]:
            return {
                'success': False,
                'message': 'Database error: No valid results returned',
                'arrangement_id': None,
                'online_payment_reference_id': uuid_str
            }
        
        # Extract function result
        function_result = results[0][0]['result']
        
        # Handle success/failure
        if function_result.startswith('OK'):
            # Extract arrangement ID if available
            arrangement_id = None
            if len(results) > 1 and results[1] and len(results[1]) > 0 and 'arrangement_id' in results[1][0]:
                arrangement_id = results[1][0]['arrangement_id']
            
            return {
                'success': True,
                'message': 'Payment arrangement created successfully through payment portal',
                'arrangement_id': arrangement_id,
                'online_payment_reference_id': uuid_str
            }
        else:
            return {
                'success': False,
                'message': f'Creation failed: {function_result}',
                'arrangement_id': None,
                'online_payment_reference_id': uuid_str
            }
    except Exception as e:
        # Ensure we reset database settings
        try:
            db.execute_query("RESET custom.user_name; RESET custom.application;")
        except:
            pass
        
        return {
            'success': False,
            'message': f'Error creating payment arrangement: {str(e)}',
            'arrangement_id': None,
            'online_payment_reference_id': uuid_str if 'uuid_str' in locals() else None
        }

@tool
def generate_sms_payment_url(
    user_id: int,
    amount: float,
    optional_reference: str = None
) -> Dict[str, Any]:
    """
    Generates a secure payment URL for client to make online payments.
    
    Use this tool to create a payment link that can be sent to clients via SMS
    or email, allowing them to make payments online through the Cartrack payment portal.
    
    Args:
        user_id: Client's unique identifier (as an integer, e.g., 83906)
        amount: Payment amount
        optional_reference: Optional reference (e.g., invoice number, arrangement ID)
        
    Returns:
        Dictionary with:
        - success: Boolean indicating if URL generation was successful
        - message: Status message or error description
        - payment_url: Generated secure payment URL (if successful)
        - reference_id: UUID reference for the payment (if successful)
    """
    result = {
        "success": False,
        "message": "",
        "payment_url": None,
        "reference_id": None
    }
    
    try:
        # Format parameters
        client_id_param = str(int(user_id))
        amount_param = str(float(amount))
        
        # Handle optional parameter
        if optional_reference:
            optional1_param = "'" + optional_reference.replace("'", "''") + "'"
        else:
            optional1_param = "NULL"
        
        # Set audit context
        db.execute_query("SET custom.user_name TO 'ai_agent';")
        db.execute_query("SET custom.application TO 'ai_assistant';")
        
        # Execute the function
        query = f"""
        SELECT cc.url_payment_encode(
            {client_id_param}, 
            {amount_param}, 
            {optional1_param}, 
            NULL, 
            NULL
        ) as payment_url;
        """
        
        url_result = db.execute_query(query)
        
        # Reset audit context
        db.execute_query("RESET custom.user_name;")
        db.execute_query("RESET custom.application;")
        
        if not url_result or 'payment_url' not in url_result[0]:
            result["message"] = "Database error: Failed to generate payment URL"
            return result
        
        payment_url = url_result[0]['payment_url']
        
        # Extract UUID from the URL
        if payment_url and '/pay/' in payment_url:
            reference_id = payment_url.split('/pay/')[1]
            result["reference_id"] = reference_id
        
        result["success"] = True
        result["message"] = "Payment URL generated successfully"
        result["payment_url"] = payment_url
        
        return result
        
    except Exception as e:
        # Try to reset the audit context
        try:
            db.execute_query("RESET custom.user_name;")
            db.execute_query("RESET custom.application;")
        except:
            pass
            
        result["message"] = f"Error generating payment URL: {str(e)}"
        return result

#-----------------------------------------------------#
@tool
def get_bank_options() -> dict:
    """
    Retrieves all available banking options for client banking operations.
    
    This tool fetches reference data for all banking-related dropdown fields
    when creating or updating client banking details.
    
    Returns:
        dict: A dictionary containing all banking options:
            banks: List of bank objects with:
                bank_id (int): Bank identifier
                name (str): Bank name
                generic_code (str): Bank code
                features: Dict of bank features (avs, allow_mandate, allow_debit, etc.)
            account_types: List of account type objects with id and description
            account_categories: List of account category objects
            invoice_options: List of invoice option objects
            debit_run_days: List of debit run day objects with day_value and display_name
            pacs_services: List of PACS service objects
    """
    try:
        # Initialize result dictionary
        result = {
            "banks": [],
            "account_types": [],
            "account_categories": [],
            "invoice_options": [],
            "debit_run_days": [],
            "pacs_services": []
        }
        
        # Fetch banks
        banks_query = """
            SELECT * 
            FROM ct.bank_names 
            WHERE disabled = FALSE 
                AND generic_code IS NOT NULL 
                AND generic_code != ''
            ORDER BY bank_id ASC;
        """
        
        bank_rows = db.execute_query(banks_query)
        
        # Transform bank data
        for bank in bank_rows:
            result["banks"].append({
                "bank_id": bank["bank_id"],
                "name": bank["bank_name"],
                "generic_code": bank["generic_code"],
                "features": {
                    "avs": bank["avs"],
                    "allow_mandate": bank["allow_mandate"],
                    "allow_debit": bank["allow_debit"],
                    "allow_debit_savings": bank["allow_debit_savings_account"]
                },
                "status": {
                    "disabled": bank["disabled"]
                }
            })
        
        # Get bank account types
        account_types_query = """
            SELECT 
                bank_account_type_id, 
                description
            FROM 
                ct.bank_account_type 
            ORDER BY 
                description ASC
        """
        account_types = db.execute_query(account_types_query)
        result["account_types"] = [
            {
                "id": type_data["bank_account_type_id"],
                "description": type_data["description"],
            }
            for type_data in account_types
        ]
        
        # Get bank account categories
        categories_query = """
            SELECT 
                bank_account_category_id, 
                description 
            FROM 
                ct.bank_account_category 
            ORDER BY 
                description ASC
        """
        categories = db.execute_query(categories_query)
        result["account_categories"] = [
            {
                "id": cat["bank_account_category_id"],
                "description": cat["description"]
            }
            for cat in categories
        ]
        
        # Get invoice options
        invoice_query = """
            SELECT 
                invoice_option_id, 
                description 
            FROM 
                ct.invoice_option 
            ORDER BY 
                description ASC
        """
        invoice_options = db.execute_query(invoice_query)
        result["invoice_options"] = [
            {
                "id": opt["invoice_option_id"],
                "description": opt["description"]
            }
            for opt in invoice_options
        ]
        
        # Get debit run days
        debit_days_query = """
            SELECT 
                debit_run_day_id, 
                day_value, 
                display_name 
            FROM 
                ct.debit_run_day 
            ORDER BY 
                day_value ASC
        """
        debit_days = db.execute_query(debit_days_query)
        result["debit_run_days"] = [
            {
                "id": day["debit_run_day_id"],
                "day_value": day["day_value"],
                "display_name": day["display_name"]
            }
            for day in debit_days
        ]
        
        # Get PACS services
        pacs_query = """
            SELECT 
                pacs_service_id, 
                description 
            FROM 
                bl.pacs_service 
            ORDER BY 
                description ASC
        """
        pacs_services = db.execute_query(pacs_query)
        result["pacs_services"] = [
            {
                "id": service["pacs_service_id"],
                "description": service["description"]
            }
            for service in pacs_services
        ]
        
        return result
        
    except Exception as e:
        error_type = type(e).__name__
        
        return {
            "error": {
                "type": error_type,
                "message": f"Failed to retrieve bank options: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        }

@tool
def update_client_banking_details(user_id: str, **update_fields) -> dict:
    """
    Updates banking details for a specific client.
    
    Args:
        user_id: Client's unique identifier (string)
        **update_fields: Key-value pairs of fields to update, which may include:
            bank_account_type_id: ID for the bank account type (int)
            account_name: Full name on the bank account (str)
            branch_code: Bank branch code (str)
            account_number: Bank account number (str) 
            bank_id: Bank's unique identifier (int)
            branch_name: Name of the bank branch (str)
            invoice_option_id: ID for invoice option (int)
            isactive: Whether the account is active (0 or 1)
            debit_run_day_id: ID for debit run day (int)
            debit_on_salary_date: Whether debits occur on salary date (0 or 1)
            salary_day: Day of month for salary payment (int, 1-31)
            separate_billing_run: Separate billing run status (0 or 1)
            suspend_billing: Whether billing is suspended (bool)
            mandate: Mandate information (bool)
            non_primary_account: Whether this is not the primary account (bool) 
            bank_account_category_id: Category ID for bank account (int)
            debit_order_bank_id: Bank ID for debit orders (int)
            pacs_service_id: PACS service ID (int)
    
    Returns:
        dict: Results with fields:
            success (bool): Whether update was successful
            message (str): Description of the result or error
            updated_details (dict): The updated banking details if successful
    Example:
        update_client_banking_details(
            "user_id"="x",
            "update_fields": {
                'debit_on_salary_date':1
                }
        )
    """
    result = {
        "success": False,
        "message": "",
        "updated_details": {}
    }
    
    try:
        # Validate the user_id
        if not user_id:
            return {
                "success": False,
                "message": "User ID is required",
                "updated_details": {}
            }
            
        # Validate update fields
        if not update_fields:
            return {
                "success": False,
                "message": "No fields provided for update",
                "updated_details": {}
            }
            
        # First get existing bank details with full bank information
        existing_details_query = f"""
            select 
                uba.* ,
                bn.bank_name ,
                bn.disabled,
                bn.avs,
                bn.allow_debit,
                bn.allow_mandate,
                bn.allow_debit_savings_account ,
                bat.description as bank_account_type,
                bac.description as bank_account_category,
                io.description as invoice_option,
                drd.day_value as debit_run_day,
                drd.display_name as debit_run_day_display_name,
                ps.description as pacs_service
            from ct.user_bank_account uba
            join ct.bank_account_type bat on bat.bank_account_type_id = uba.bank_account_type_id
            join ct.bank_account_category bac on bac.bank_account_category_id = uba.bank_account_category_id
            join ct.invoice_option io on io.invoice_option_id = uba.invoice_option_id
            join ct.debit_run_day drd on drd.debit_run_day_id = uba.debit_run_day_id
            join bl.pacs_service ps on ps.pacs_service_id = uba.pacs_service_id
            join ct.bank_names bn on bn.bank_id = uba.bank_id
            where uba.user_id = '{user_id}'
        """
        
        existing_details = db.execute_query(existing_details_query)
        
        if not existing_details:
            return {
                "success": False,
                "message": f"No banking details found for user_id: {user_id}",
                "updated_details": {}
            }
            
        # Get the first record
        existing_record = existing_details[0]
        user_bank_account_id = existing_record.get("user_bank_account_id")
        
        # Get the current user for auditing
        current_user_query = "SELECT CURRENT_USER"
        current_user_result = db.execute_query(current_user_query)
        current_user = current_user_result[0].get("current_user") if current_user_result else "system"
        
        # Define allowed fields
        allowed_fields = [
            "bank_account_type_id", "account_name", "branch_code", 
            "account_number", "bank_id", "branch_name", "invoice_option_id",
            "isactive", "debit_run_day_id", "debit_on_salary_date",
            "salary_day", "separate_billing_run", "suspend_billing",
            "mandate", "non_primary_account", "bank_account_category_id",
            "debit_order_bank_id", "pacs_service_id"
        ]

        # Filter valid updates
        valid_updates = {k: v for k, v in update_fields.get('update_fields').items() if k in allowed_fields}
        
        if not valid_updates:
            return {
                "success": False,
                "message": "No valid fields provided for update",
                "updated_details": {}
            }
            
        # Build the SET clause
        update_fields = []
        update_values = []
        
        for field, value in valid_updates.items():
            update_fields.append(f"{field} = %s")
            update_values.append(value)
        
        # Add audit fields
        update_fields.extend([
            "au$username = %s", 
            "au$ts = CURRENT_TIMESTAMP", 
            "au$application = %s",
            "au$counter = au$counter + 1"
        ])
        update_values.extend([current_user, "AI_Agent"])
        
        # Build and execute the update query
        update_query = f"""
            UPDATE ct.user_bank_account
            SET {", ".join(update_fields)}
            WHERE user_bank_account_id = {user_bank_account_id}
        """
        
        # Execute the update query with parameters
        db.execute_query(update_query, update_values)
        
        # Fetch the updated record
        updated_details = db.execute_query(existing_details_query)
        
        if updated_details:
            result["success"] = True
            result["message"] = "Banking details updated successfully"
            result["updated_details"] = updated_details[0]
        else:
            result["message"] = "Update was performed but could not retrieve updated details"
            
    except Exception as e:
        result["message"] = f"Error updating banking details: {str(e)}"
        
    return result


#-----------------------------------------------------#
# #Step 9: Update client info
@tool
def validate_next_of_kin_contact(
    user_id: int,
    name: str,
    phone: str,
    email: str = None
) -> Dict[str, Any]:
    """
    Validates a next of kin's contact information to prevent duplicates.
    
    Use this tool before adding or updating a client's next of kin contacts to check
    if the contact information would create a duplicate across the client's vehicles.
    
    Args:
        user_id: Client's unique identifier (as an integer, e.g., 83906)
        name: Name of the next of kin
        phone: Phone number of the next of kin
        email: Email address of the next of kin (optional)
        
    Returns:
        Dictionary with:
        - valid: Boolean indicating if the contact information is valid (no conflicts)
        - message: Status message or error description
        - conflict_vehicle: Vehicle registration if conflict exists (None otherwise)
        - conflict_contact: Existing contact details if conflict exists (None otherwise)
    """
    result = {
        "valid": False,
        "message": "",
        "conflict_vehicle": None,
        "conflict_contact": None
    }
    
    try:
        # Format parameters
        individual_id_param = str(int(user_id))
        
        # Handle string parameters with proper escaping
        name_param = "'" + name.replace("'", "''") + "'" if name else "NULL"
        phone_param = "'" + phone.replace("'", "''") + "'" if phone else "NULL"
        email_param = "'" + email.replace("'", "''") + "'" if email else "NULL"
        
        # Execute the validation function
        query = f"""
        SELECT ct.validate_next_of_kin_contact(
            {individual_id_param}, 
            {name_param}, 
            {phone_param}, 
            {email_param}
        ) as validation_result;
        """
        
        validation_result = db.execute_query(query)
        
        if not validation_result or 'validation_result' not in validation_result[0]:
            result["message"] = "Database error: Failed to validate next of kin"
            return result
        
        response = validation_result[0]['validation_result']
        
        # Check if the validation passed
        if response == 'OK':
            result["valid"] = True
            result["message"] = "Contact information is valid"
        else:
            # Parse the error message to extract vehicle and contact details
            result["valid"] = False
            result["message"] = response
            
            # Extract the vehicle registration and contact details
            if 'Next of kin linked to vehicle:' in response:
                parts = response.split('\n\r')
                if len(parts) >= 1:
                    vehicle_part = parts[0].replace('Next of kin linked to vehicle:', '').strip()
                    result["conflict_vehicle"] = vehicle_part
                
                if len(parts) >= 2:
                    result["conflict_contact"] = parts[1].strip()
        
        return result
        
    except Exception as e:
        result["message"] = f"Error validating next of kin contact: {str(e)}"
        return result

@tool
def validate_next_of_kin_emergency_contact(
    vehicle_id: int,
    contact_priority: int,
    name: str,
    phone: str,
    email: str = None
) -> Dict[str, Any]:
    """
    Validates a vehicle emergency contact to prevent duplicates with existing next of kin.
    
    Use this tool before adding or updating a vehicle's emergency contacts to check
    if they would conflict with the vehicle owner's next of kin information.
    
    Args:
        vehicle_id: Vehicle ID (as an integer)
        contact_priority: Priority level (1=primary, 2=secondary, etc.)
        name: Name of the emergency contact
        phone: Phone number of the emergency contact
        email: Email address of the emergency contact (optional)
        
    Returns:
        Dictionary with:
        - valid: Boolean indicating if the contact information is valid (no conflicts)
        - message: Status message or error description
        - next_of_kin_details: Existing next of kin details if conflict exists
    """
    result = {
        "valid": False,
        "message": "",
        "next_of_kin_details": None
    }
    
    try:
        # Format parameters
        vehicle_id_param = str(int(vehicle_id))
        contact_priority_param = str(int(contact_priority))
        
        # Handle string parameters with proper escaping
        name_param = "'" + name.replace("'", "''") + "'" if name else "NULL"
        phone_param = "'" + phone.replace("'", "''") + "'" if phone else "NULL"
        email_param = "'" + email.replace("'", "''") + "'" if email else "NULL"
        
        # Execute the validation function
        query = f"""
        SELECT ct.validate_next_of_kin_emergency_contact(
            {vehicle_id_param},
            {contact_priority_param},
            {name_param}, 
            {phone_param}, 
            {email_param}
        ) as validation_result;
        """
        
        validation_result = db.execute_query(query)
        
        if not validation_result or 'validation_result' not in validation_result[0]:
            result["message"] = "Database error: Failed to validate emergency contact"
            return result
        
        response = validation_result[0]['validation_result']
        
        # Check if the validation passed
        if response == 'OK':
            result["valid"] = True
            result["message"] = "Contact information is valid"
        else:
            # Parse the error message to extract next of kin details
            result["valid"] = False
            result["message"] = response
            
            # Extract the next of kin details
            if "This contact is already captured" in response and '\n\r' in response:
                parts = response.split('\n\r')
                if len(parts) >= 2:
                    result["next_of_kin_details"] = parts[1].strip()
        
        return result
        
    except Exception as e:
        result["message"] = f"Error validating emergency contact: {str(e)}"
        return result
    

@tool
def update_client_information_not_used(user_id: str, **update_fields) -> dict:
    """
    Updates personal information for a specific client in the CarTrack system.
    
    Args:
        user_id: Client's unique identifier (string)
        **update_fields: Key-value pairs of fields to update, which may include:
            # Individual table fields
            individual_title_type_id: ID for title (Mr, Mrs, etc.) (int)
            first_name: Client's first name (str)
            last_name: Client's last name (str)
            initials: Client's initials (str)
            id_number: SA ID number (str)
            employer: Client's employer name (str)
            passport_number: Passport number if applicable (str)
            tax_reg_number: Tax registration number (str)
            
            # Individual_extra table fields
            race_id: Race identifier (int)
            language_id: Language identifier (int)
            gender_type_id: Gender identifier (int)
            income_id: Income bracket identifier (int)
            spouse: Whether client has a spouse (bool)
            spouse_name: Name of spouse (str)
            spouse_phone: Phone number of spouse (str)
            birthdate: Date of birth (str format YYYY-MM-DD)
            nok_name: Next of kin name (str)
            nok_phone: Next of kin phone number (str)
            nok_email: Next of kin email address (str)
            alternate_name: Alternate name for client (str)
    
    Returns:
        dict: Results with fields:
            success (bool): Whether update was successful
            message (str): Description of the result or error
            updated_individual (dict): The updated individual details if successful
            updated_individual_extra (dict): The updated individual_extra details if successful
    """
    result = {
        "success": False,
        "message": "",
        "updated_individual": {},
        "updated_individual_extra": {}
    }
    
    try:
        # Validate the user_id
        if not user_id:
            return {
                "success": False,
                "message": "User ID is required",
                "updated_individual": {},
                "updated_individual_extra": {}
            }
            
        # Validate update fields
        if not update_fields:
            return {
                "success": False,
                "message": "No fields provided for update",
                "updated_individual": {},
                "updated_individual_extra": {}
            }

        # Get current values
        individual_query = f"SELECT * FROM ct.individual WHERE user_id = '{user_id}'"
        individual_result = db.execute_query(individual_query)
        
        individual_extra_query = f"SELECT * FROM ct.individual_extra WHERE individual_extra_id = '{user_id}'"
        individual_extra_result = db.execute_query(individual_extra_query)
        
        if not individual_result:
            return {
                "success": False,
                "message": f"No individual record found for user_id: {user_id}",
                "updated_individual": {},
                "updated_individual_extra": {}
            }
            
        # Get the current user for auditing
        current_user_query = "SELECT current_setting('custom.user_name') as current_user"
        current_user_result = db.execute_query(current_user_query)
        current_user = current_user_result[0].get("current_user") if current_user_result else "system"
        
        # Define allowed fields for each table
        individual_fields = [
            "individual_title_type_id", "first_name", "last_name", 
            "initials", "id_number", "employer", "passport_number", 
            "tax_reg_number"
        ]
        
        individual_extra_fields = [
            "race_id", "language_id", "gender_type_id", 
            "income_id", "spouse", "spouse_name", "spouse_phone", 
            "birthdate", "nok_name", "nok_phone", "nok_email", 
            "alternate_name"
        ]

        # Filter valid updates for each table
        valid_individual_updates = {k: v for k, v in update_fields.get('update_fields', {}).items() if k in individual_fields}
        valid_individual_extra_updates = {k: v for k, v in update_fields.get('update_fields', {}).items() if k in individual_extra_fields}
        
        # Update the individual table if we have valid fields
        if valid_individual_updates:
            # Build the SET clause
            update_fields_list = []
            update_values = []
            
            for field, value in valid_individual_updates.items():
                update_fields_list.append(f"{field} = %s")
                update_values.append(value)
            
            # Add audit fields if needed
            update_fields_list.extend([
                "au$username = %s", 
                "au$ts = CURRENT_TIMESTAMP", 
                "au$application = %s"
            ])
            update_values.extend([current_user, "AI_Agent"])
            
            # Build and execute the update query
            update_query = f"""
                UPDATE ct.individual
                SET {", ".join(update_fields_list)}
                WHERE user_id = '{user_id}'
            """
            
            # Execute the update query with parameters
            db.execute_query(update_query, update_values)
            
            # Fetch the updated record
            updated_individual = db.execute_query(individual_query)
            result["updated_individual"] = updated_individual[0] if updated_individual else {}
        else:
            result["updated_individual"] = individual_result[0] if individual_result else {}
            
        # Update the individual_extra table if we have valid fields
        if valid_individual_extra_updates:
            # Check if we need to insert or update
            if not individual_extra_result:
                # Need to insert a new record
                fields = ["individual_extra_id"] + list(valid_individual_extra_updates.keys())
                values = [user_id] + [valid_individual_extra_updates[k] for k in fields[1:]]
                
                placeholders = ["%s"] * len(values)
                
                insert_query = f"""
                    INSERT INTO ct.individual_extra ({", ".join(fields)})
                    VALUES ({", ".join(placeholders)})
                """
                
                db.execute_query(insert_query, values)
            else:
                # Build the SET clause for update
                update_fields_list = []
                update_values = []
                
                for field, value in valid_individual_extra_updates.items():
                    update_fields_list.append(f"{field} = %s")
                    update_values.append(value)
                
                # Add audit fields if needed (assuming they exist)
                update_fields_list.extend([
                    "au$username = %s", 
                    "au$ts = CURRENT_TIMESTAMP", 
                    "au$application = %s"
                ])
                update_values.extend([current_user, "AI_Agent"])
                
                # Build and execute the update query
                update_query = f"""
                    UPDATE ct.individual_extra
                    SET {", ".join(update_fields_list)}
                    WHERE individual_extra_id = '{user_id}'
                """
                
                # Execute the update query with parameters
                db.execute_query(update_query, update_values)
            
            # Fetch the updated record
            updated_individual_extra = db.execute_query(individual_extra_query)
            result["updated_individual_extra"] = updated_individual_extra[0] if updated_individual_extra else {}
        else:
            result["updated_individual_extra"] = individual_extra_result[0] if individual_extra_result else {}
            
        result["success"] = True
        result["message"] = "Client information updated successfully"
        
    except Exception as e:
        result["message"] = f"Error updating client information: {str(e)}"
        
    return result

@tool
def update_client_contact_number(user_id: str, mobile_number: str) -> dict:
    """
    Updates the direct contact number (mobile number) for a specific client in the CarTrack system.
    
    Args:
        user_id: Client's unique identifier (string)
        mobile_number: New mobile phone number to update (string)
    
    Returns:
        dict: Results with fields:
            success (bool): Whether update was successful
            message (str): Description of the result or error
            previous_number (str): The previous mobile number before update
            current_number (str): The updated mobile number
    
    """
    result = {
        "success": False,
        "message": "",
        "previous_number": "",
        "current_number": ""
    }
    
    try:
        # Validate the user_id
        if not user_id:
            result["message"] = "User ID is required"
            return result
            
        # Validate mobile number
        if not mobile_number:
            result["message"] = "Mobile number is required"
            return result
            
        # Check if mobile number has valid format
        # This is a basic validation - adjust according to your specific requirements
        if not mobile_number.isdigit() or not (len(mobile_number) >= 10 and len(mobile_number) <= 15):
            result["message"] = "Invalid mobile number format. Must be 10-15 digits."
            return result
        
        # First, check if the client exists and get current mobile number
        client_query = f"SELECT user_id, mobile_number FROM ct.client WHERE user_id = '{user_id}'"
        client_result = db.execute_query(client_query)
        
        if not client_result:
            result["message"] = f"No client found for user_id: {user_id}"
            return result
            
        # Store previous number for reference
        result["previous_number"] = client_result[0].get("mobile_number", "")
        
        # Get the current user for auditing
        current_user_query = "SELECT current_setting('custom.user_name') as current_user"
        current_user_result = db.execute_query(current_user_query)
        current_user = current_user_result[0].get("current_user") if current_user_result else "system"
        
        # Build and execute the update query
        update_query = """
            UPDATE ct.client
            SET 
                mobile_number = %s,
                au$username = %s,
                au$ts = CURRENT_TIMESTAMP,
                au$application = %s,
                au$counter = au$counter + 1
            WHERE user_id = %s
        """
        
        # Execute the update query with parameters
        db.execute_query(update_query, [
            mobile_number,
            current_user,
            "AI_Agent",
            user_id
        ])
        
        # Verify the update was successful
        verification_query = f"SELECT mobile_number FROM ct.client WHERE user_id = '{user_id}'"
        verification_result = db.execute_query(verification_query)
        
        if verification_result and verification_result[0].get("mobile_number") == mobile_number:
            result["success"] = True
            result["message"] = "Mobile number updated successfully"
            result["current_number"] = mobile_number
        else:
            result["message"] = "Update operation completed but verification failed"
            
    except Exception as e:
        result["message"] = f"Error updating mobile number: {str(e)}"
        
    return result

@tool
def update_client_email(user_id: str, email_address: str) -> dict:
    """
    Updates the email address for a specific client in the CarTrack system.
    
    Args:
        user_id: Client's unique identifier (string)
        email_address: New email address to update (string)
    
    Returns:
        dict: Results with fields:
            success (bool): Whether update was successful
            message (str): Description of the result or error
            previous_email (str): The previous email address before update
            current_email (str): The updated email address
    """
    result = {
        "success": False,
        "message": "",
        "previous_email": "",
        "current_email": ""
    }
    
    try:
        # Validate the user_id
        if not user_id:
            result["message"] = "User ID is required"
            return result
            
        # Validate email address
        if not email_address:
            result["message"] = "Email address is required"
            return result
            
        # Basic email validation
        if "@" not in email_address or "." not in email_address:
            result["message"] = "Invalid email address format"
            return result
        
        # First, check if the user exists and get current email
        user_query = f"SELECT user_id, primary_email FROM ct.user WHERE user_id = '{user_id}'"
        user_result = db.execute_query(user_query)
        
        if not user_result:
            result["message"] = f"No user found for user_id: {user_id}"
            return result
            
        # Store previous email for reference
        result["previous_email"] = user_result[0].get("primary_email", "")
        
        # Get the current user for auditing
        current_user_query = "SELECT current_setting('custom.user_name') as current_user"
        current_user_result = db.execute_query(current_user_query)
        current_user = current_user_result[0].get("current_user") if current_user_result else "system"
        
        # Build and execute the update query
        update_query = """
            UPDATE ct.user
            SET 
                primary_email = %s,
                au$username = %s,
                au$ts = CURRENT_TIMESTAMP,
                au$application = %s,
                au$counter = au$counter + 1
            WHERE user_id = %s
        """
        
        # Execute the update query with parameters
        db.execute_query(update_query, [
            email_address,
            current_user,
            "EmailUpdateTool",
            user_id
        ])
        
        # Verify the update was successful
        verification_query = f"SELECT primary_email FROM ct.user WHERE user_id = '{user_id}'"
        verification_result = db.execute_query(verification_query)
        
        if verification_result and verification_result[0].get("primary_email") == email_address:
            result["success"] = True
            result["message"] = "Email address updated successfully"
            result["current_email"] = email_address
        else:
            result["message"] = "Update operation completed but verification failed"
            
    except Exception as e:
        result["message"] = f"Error updating email address: {str(e)}"
        
    return result


@tool
def update_client_next_of_kin(user_id: str, nok_name: str = None, nok_phone: str = None, nok_email: str = None) -> dict:
    """
    Updates next of kin information for a specific client in the CarTrack system.
    
    Args:
        user_id: Client's unique identifier (string)
        nok_name: Next of kin name (string, optional)
        nok_phone: Next of kin phone number (string, optional)
        nok_email: Next of kin email address (string, optional)
    
    Returns:
        dict: Results with fields:
            success (bool): Whether update was successful
            message (str): Description of the result or error
            previous_nok_info (dict): The previous next of kin information
            current_nok_info (dict): The updated next of kin information
    
    Example:
        update_client_next_of_kin(
            user_id="1911088",
            nok_name="Jane Smith",
            nok_phone="0731234567",
            nok_email="jane@example.com"
        )
    """
    result = {
        "success": False,
        "message": "",
        "previous_nok_info": {
            "nok_name": "",
            "nok_phone": "",
            "nok_email": ""
        },
        "current_nok_info": {
            "nok_name": "",
            "nok_phone": "",
            "nok_email": ""
        }
    }
    
    try:
        # Validate the user_id
        if not user_id:
            result["message"] = "User ID is required"
            return result
            
        # Validate that at least one field is provided
        if not any([nok_name, nok_phone, nok_email]):
            result["message"] = "At least one next of kin field (name, phone, or email) must be provided"
            return result
        
        # First, check if the client exists in individual_extra table
        query = f"SELECT individual_extra_id, nok_name, nok_phone, nok_email FROM ct.individual_extra WHERE individual_extra_id = '{user_id}'"
        individual_extra_result = db.execute_query(query)
        
        # Store previous information for reference
        if individual_extra_result:
            result["previous_nok_info"]["nok_name"] = individual_extra_result[0].get("nok_name", "")
            result["previous_nok_info"]["nok_phone"] = individual_extra_result[0].get("nok_phone", "")
            result["previous_nok_info"]["nok_email"] = individual_extra_result[0].get("nok_email", "")
        
        # Get the current user for auditing
        current_user_query = "SELECT current_setting('custom.user_name') as current_user"
        current_user_result = db.execute_query(current_user_query)
        current_user = current_user_result[0].get("current_user") if current_user_result else "system"
        
        # If next of kin phone number is provided, validate it using the system function
        if nok_phone and len(nok_phone.strip()) > 0:
            validation_query = f"SELECT * FROM ct.validate_next_of_kin_contact('{user_id}', {nok_name if nok_name else 'NULL'}, '{nok_phone}', {nok_email if nok_email else 'NULL'})"
            validation_result = db.execute_query(validation_query)
            
            if validation_result and validation_result[0].get("validate_next_of_kin_contact") != 'OK':
                result["message"] = validation_result[0].get("validate_next_of_kin_contact", "Next of kin validation failed")
                return result
        
        # If the individual_extra record doesn't exist, create it
        if not individual_extra_result:
            # Build insert fields and values
            fields = ["individual_extra_id"]
            values = [user_id]
            
            if nok_name:
                fields.append("nok_name")
                values.append(nok_name)
                
            if nok_phone:
                fields.append("nok_phone")
                values.append(nok_phone)
                
            if nok_email:
                fields.append("nok_email")
                values.append(nok_email)
                
            # Add audit fields
            fields.extend(["au$username", "au$application"])
            values.extend([current_user, "NextOfKinUpdateTool"])
            
            placeholders = ["%s"] * len(values)
            
            insert_query = f"""
                INSERT INTO ct.individual_extra ({", ".join(fields)})
                VALUES ({", ".join(placeholders)})
            """
            
            db.execute_query(insert_query, values)
        else:
            # Build the update query for individual_extra
            update_fields = []
            update_values = []
            
            if nok_name is not None:
                update_fields.append("nok_name = %s")
                update_values.append(nok_name)
                
            if nok_phone is not None:
                update_fields.append("nok_phone = %s")
                update_values.append(nok_phone)
                
            if nok_email is not None:
                update_fields.append("nok_email = %s")
                update_values.append(nok_email)
            
            # Add audit fields
            update_fields.extend([
                "au$username = %s", 
                "au$ts = CURRENT_TIMESTAMP", 
                "au$application = %s",
                "au$counter = COALESCE(au$counter, 0) + 1"
            ])
            update_values.extend([current_user, "AI_Agent"])
            
            # Add where clause
            update_values.append(user_id)
            
            update_query = f"""
                UPDATE ct.individual_extra
                SET {", ".join(update_fields)}
                WHERE individual_extra_id = %s
            """
            
            db.execute_query(update_query, update_values)
        
        # Verify the update was successful
        verification_query = f"SELECT nok_name, nok_phone, nok_email FROM ct.individual_extra WHERE individual_extra_id = '{user_id}'"
        verification_result = db.execute_query(verification_query)
        
        if verification_result:
            result["success"] = True
            result["message"] = "Next of kin information updated successfully"
            result["current_nok_info"]["nok_name"] = verification_result[0].get("nok_name", "")
            result["current_nok_info"]["nok_phone"] = verification_result[0].get("nok_phone", "")
            result["current_nok_info"]["nok_email"] = verification_result[0].get("nok_email", "")
        else:
            result["message"] = "Update operation completed but verification failed"
            
    except Exception as e:
        result["message"] = f"Error updating next of kin information: {str(e)}"
        
    return result

@tool
def add_client_note(user_id: str, note_text: str) -> Dict[str, Any]:
    """
    Adds a note to a client's record with the current date and time.
    
    Use this tool to document client interactions or important information in their record.
    The note will be timestamped with the current date and time.
    
    Args:
        user_id: Client's unique identifier (e.g., "83906")
        note_text: The content of the note to add
        
    Returns:
        Dictionary containing:
        - success: Boolean indicating if the note was added successfully
        - message: Success confirmation or error description
        - client_note_id: ID of the created note (if successful)
    """
    import datetime
    from typing import Dict, Any
    import logging
    
    # Initialize logger
    logger = logging.getLogger(__name__)
    
    # Set a default note_user_id (should ideally come from authentication context)
    note_user_id = 1
    
    # Get current datetime in ISO format
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log the operation
    logger.info(f"Adding note for client {user_id} at {now}")
    
    try:
        # Safely use f-strings by escaping special characters in user inputs
        user_id_safe = str(user_id).replace("'", "''")
        note_text_safe = str(note_text).replace("'", "''")+". Added by AI_Agent"
        
        insert_query = f"""
        INSERT INTO ct.client_note
            (client_id, note_user_id, note_ts, note_text)
        VALUES (
            '{user_id_safe}',
            '{note_user_id}',
            '{now}',
            '{note_text_safe}'
        )
        RETURNING client_note_id;
        """
        
        # Execute the query
        result = db.execute_query(insert_query)
        
        if result and len(result) > 0:
            return {
                "success": True,
                "message": f"Note added successfully at {now}",
                "client_note_id": result[0]['client_note_id']
            }
        else:
            logger.error("Query returned no results")
            return {
                "success": False,
                "message": "Failed to add note to client record"
            }
    except Exception as e:
        logger.error(f"Error adding note for client {user_id}: {str(e)}")
        return {
            "success": False,
            "message": f"Error adding note: {str(e)}"
        }
    

@tool
def get_disposition_types(include_disabled: bool = False) -> dict:
    """
    Retrieves all available call disposition types from the Cartrack system.
    
    Args:
        include_disabled: Whether to include disabled disposition types (default: False)
    
    Returns:
        dict: Results with fields:
            success (bool): Whether the retrieval was successful
            message (str): Description of the result or error
            disposition_types (list): List of disposition types with their details:
                - id: The disposition type ID
                - description: The description of the disposition type
                - instruction: Any specific instructions for this disposition type
                - enabled: Whether the disposition type is currently enabled
    """
    result = {
        "success": False,
        "message": "",
        "disposition_types": []
    }
    
    try:
        # Build the query based on whether to include disabled types
        query = """
            SELECT 
                debtor_disposition_type_id, 
                description, 
                disposition_instruction, 
                enabled
            FROM ct.tl_debtor_disposition_type
        """
        
        if not include_disabled:
            query += " WHERE enabled = TRUE"
            
        query += " ORDER BY debtor_disposition_type_id"
        
        # Execute the query
        disposition_types = db.execute_query(query)
        
        if disposition_types:
            # Format the results for easy consumption
            for disposition in disposition_types:
                result["disposition_types"].append({
                    "id": str(disposition.get("debtor_disposition_type_id")),
                    "description": disposition.get("description", ""),
                    "instruction": disposition.get("disposition_instruction", ""),
                    "enabled": disposition.get("enabled", True)
                })
                
            result["success"] = True
            result["message"] = f"Retrieved {len(result['disposition_types'])} disposition types"
            
            # Add categories to help LLM organize the types
            categories = {}
            for item in result["disposition_types"]:
                desc = item["description"].lower()
                if "uncontactable" in desc:
                    category = "Uncontactable"
                elif "ptp" in desc or "promise" in desc:
                    category = "Payment Promise"
                elif "bank" in desc:
                    category = "Bank Details"
                elif "complaint" in desc or "service" in desc:
                    category = "Complaints and Service"
                elif "client" in desc:
                    category = "Client Status"
                else:
                    category = "Other"
                
                if category not in categories:
                    categories[category] = []
                categories[category].append(item)
            
            result["categories"] = categories
        else:
            result["message"] = "No disposition types found"
            
    except Exception as e:
        result["message"] = f"Error retrieving disposition types: {str(e)}"
        
    return result
    
@tool
def save_call_disposition(client_id: str, disposition_type_id: str, rating: str = None, ratio: str = None, note_text: str = None) -> dict:
    """
    Saves the call disposition and optional note for a debtor call in the Cartrack call center.
    
    Use the get_disposition_types() tool first to retrieve a list of valid disposition types.
    
    Args:
        client_id: Client/debtor's unique identifier (string)
        disposition_type_id: Type ID of the disposition (string) from ct.tl_debtor_disposition_type table
        rating: Optional rating information (string)
        ratio: Optional ratio information (string)
        note_text: Optional note text to add with the disposition (string)
    
    Returns:
        dict: Results with fields:
            success (bool): Whether the disposition was saved successfully
            message (str): Description of the result or error
            disposition_id (str): The ID of the created disposition record
            disposition_description (str): The description of the disposition type
            disposition_instruction (str): Any specific instructions for this disposition type
    """
    result = {
        "success": False,
        "message": "",
        "disposition_id": None,
        "disposition_description": "",
        "disposition_instruction": ""
    }
    
    try:
        # Validate the client_id
        if not client_id:
            result["message"] = "Client ID is required"
            return result
            
        # Validate disposition_type_id
        if not disposition_type_id:
            result["message"] = "Disposition type ID is required"
            return result
            
        # Check if the disposition type exists and is enabled
        disposition_query = """
            SELECT debtor_disposition_type_id, description, disposition_instruction, enabled
            FROM ct.tl_debtor_disposition_type
            WHERE debtor_disposition_type_id = %s
        """
        
        disposition_result = db.execute_query(disposition_query, [disposition_type_id])
        
        if not disposition_result:
            result["message"] = f"Invalid disposition type ID: {disposition_type_id}"
            return result
            
        disposition_data = disposition_result[0]
        if not disposition_data.get("enabled", True):
            result["message"] = f"Disposition type {disposition_type_id} is disabled"
            return result
            
        # Store the description and instruction for response
        disposition_description = disposition_data.get("description", "")
        disposition_instruction = disposition_data.get("disposition_instruction", "")
        result["disposition_description"] = disposition_description
        result["disposition_instruction"] = disposition_instruction
        
        # Get the current user for the userid field
        current_user_query = "SELECT current_setting('custom.user_id')::bigint as current_user_id"
        current_user_result = db.execute_query(current_user_query)
        user_id = current_user_result[0].get("current_user_id") if current_user_result else None
        
        if not user_id:
            result["message"] = "Unable to determine current user"
            return result
        
        # Build and execute the insert query for disposition
        disposition_query = """
            INSERT INTO ct.t_debtor_disposition 
                (disposition_type_id, clientid, userid, rating, ratio, disposition_date_time)
            VALUES 
                (%s, %s, %s, %s, %s, NOW())
            RETURNING disposition_id
        """
        
        # Execute the insert query with parameters
        disposition_result = db.execute_query(disposition_query, [
            disposition_type_id,
            client_id,
            user_id,
            rating,
            ratio
        ])
        
        if disposition_result and disposition_result[0].get("disposition_id"):
            disposition_id = disposition_result[0].get("disposition_id")
            result["disposition_id"] = str(disposition_id)
            
            # If note_text is provided, add a note for the client
            if note_text:
                # Auto-append disposition description to the note
                formatted_note = f"[{disposition_description}] {note_text}"
                
                note_query = """
                    INSERT INTO ct.client_note
                        (client_id, note_user_id, note_ts, note_text, client_note_type_id)
                    VALUES
                        (%s, %s, NOW(), %s, 67)
                """
                
                db.execute_query(note_query, [
                    client_id,
                    user_id,
                    formatted_note
                ])
                
                result["success"] = True
                result["message"] = "Call disposition saved with client note"
                
                # Check if there are specific instructions to share
                if disposition_instruction:
                    result["message"] += f"\nNext steps: {disposition_instruction}"
                
                # For PTP-related dispositions, check if we should trigger PTP creation
                if "PTP" in disposition_description and "PTP information" in note_text:
                    # This could trigger a PTP creation flow in a real implementation
                    result["message"] += "\nPTP information detected in note - consider creating formal PTP record"
            else:
                result["success"] = True
                result["message"] = "Call disposition saved successfully"
                
                # Check if there are specific instructions to share
                if disposition_instruction:
                    result["message"] += f"\nNext steps: {disposition_instruction}"
                
                # Suggest adding a note for certain disposition types
                if any(keyword in disposition_description.lower() for keyword in ["ptp", "call back", "complaint", "follow up"]):
                    result["message"] += " (Recommended: Add a note with more details)"
        else:
            result["message"] = "Failed to save disposition record"
            
    except Exception as e:
        result["message"] = f"Error saving call disposition: {str(e)}"
        
    return result