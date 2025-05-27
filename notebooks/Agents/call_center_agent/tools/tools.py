"""
Tools for the Call Center AI Agent.

This module contains all the tools used by the Call Center AI Agent
to interact with external systems and services during debt collection calls.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import random
import re
from langchain_core.tools import tool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------
# Database interaction tools
# -------------------------------------------------------------------------------------
@tool
def get_debtor_info(user_id: int) -> Dict[str, Any]:
    """
    Retrieve basic debtor information from the database.
    
    Args:
        user_id: The client's unique identifier
            
    Returns:
        Dictionary with client details
    """
    # In a real implementation, this would call: ct.get_debtor_info(:user_id)
    # For simulation, return mock data
    return {
        "user_id": user_id,
        "full_name": "John Doe",  # This would come from real DB
        "username": "jdoe123",
        "status": "Active",
        "retrieved_from": "ct.get_debtor_info"
    }
            
@tool
def get_client_detail_information(user_id: int) -> Dict[str, Any]:
    """
    Retrieve detailed client information including personal and vehicle details.
    
    Args:
        user_id: The client's unique identifier
        
    Returns:
        Dictionary with comprehensive client details
    """
    # In a real implementation, this would call: ct.sp_client_information(user_id)
    # For simulation, create a more detailed response
    
    # Extract birth date from ID number if available
    id_number = "7001015001081"  # Simulated ID number
    birth_date = None
    if id_number and len(id_number) >= 6:
        # Format: YYMMDD in first 6 digits
        birth_date = f"19{id_number[:2]}-{id_number[2:4]}-{id_number[4:6]}"
        
    # Simulate vehicle details
    vehicle_details = [
        {
            "registration": "ABC123GP",
            "make": "Toyota",
            "model": "Corolla",
            "color": "Silver",
            "chassis_nr": f"VIN{random.randint(10000, 99999)}"  # Simulate VIN number
        }
    ]
        
    return {
        "user_id": user_id,
        "personal": {
            "full_name": "John Doe",
            "id_number": id_number,
            "birth_date": birth_date,
            "email": "john.doe@example.com",
            "residential_address": "123 Main St, Johannesburg",
            "postal_address": "PO Box 456, Johannesburg"
        },
        "contact": {
            "mobile": "+27123456789",
            "home": "",
            "work": "+27111234567"
        },
        "vehicles": vehicle_details,
        "next_of_kin": {
            "name": "Jane Doe",
            "relationship": "Spouse",
            "contact": "+27987654321"
        },
        "retrieved_from": "ct.sp_client_information"
    }

@tool
def get_debtors_client_info(user_id: int) -> Dict[str, Any]:
    """
    Retrieve client information specifically formatted for debt collection.
    
    Args:
        user_id: The client's unique identifier
        
    Returns:
        Dictionary with client details relevant for collections
    """
    # In a real implementation, this would call: ct.sp_debtors_client_info(:user_id)
    
    # Simulate collections-specific information
    return {
        "user_id": user_id,
        "full_name": "John Doe",
        "contact_number": "+27123456789",
        "email": "john.doe@example.com",
        "collections_data": {
            "last_contact_date": "2024-03-15",
            "contact_count": 3,
            "prior_resolution_attempts": 2,
            "collector_notes": "Client has been difficult to reach."
        },
        "retrieved_from": "ct.sp_debtors_client_info"
    }

@tool
def get_client_age_analysis(user_id: int, current_date: str = None) -> Dict[str, Any]:
    """
    Retrieve age analysis of client's account (overdue amounts by aging periods).
    
    Args:
        user_id: The client's unique identifier
        current_date: Current date for age calculation (default: today)
        
    Returns:
        Dictionary with age analysis of overdue amounts
    """
    # In a real implementation, this would call: ct.sp_debtors_age(:user_id, current_date)
    if not current_date:
        current_date = datetime.now().strftime("%Y-%m-%d")
        
    # Simulate aging buckets
    total_outstanding = 1850.00
    current = total_outstanding * 0.2
    days_30 = total_outstanding * 0.3
    days_60 = total_outstanding * 0.4
    days_90 = total_outstanding * 0.1
    days_120 = 0
        
    return {
        "user_id": user_id,
        "as_of_date": current_date,
        "total_outstanding": f"R {total_outstanding:.2f}",
        "aging": {
            "current": f"R {current:.2f}",
            "30_days": f"R {days_30:.2f}",
            "60_days": f"R {days_60:.2f}",
            "90_days": f"R {days_90:.2f}",
            "120_plus_days": f"R {days_120:.2f}"
        },
        "retrieved_from": "ct.sp_debtors_age"
    }

@tool
def get_client_account_state(user_id: int) -> Dict[str, Any]:
    """
    Retrieve the current account state/status of a client.
    
    Args:
        user_id: The client's unique identifier
        
    Returns:
        Dictionary with account state information
    """
    # In a real implementation, this would call: ct.sp_get_client_account_state
    
    # Simulate account state
    outstanding = 1850.00
    state = "Arrears"
    severity = "Moderate"
        
    return {
        "user_id": user_id,
        "account_state": state,
        "severity": severity,
        "days_overdue": random.randint(30, 90),
        "suspension_status": "Active" if outstanding < 1000 else "Suspended",
        "retrieved_from": "ct.sp_get_client_account_state"
    }

@tool
def get_contract_details(user_id: int) -> Dict[str, Any]:
    """
    Retrieve details about the client's contract and services.
    
    Args:
        user_id: The client's unique identifier
        
    Returns:
        Dictionary with contract information
    """
    # In a real implementation, this would access: contact_details_valid_state table
    
    # Simulate contract
    contract_start = datetime(2021, 1, 1).strftime("%Y-%m-%d")
    contract = {
        "contract_id": 10000 + random.randint(1000, 9999),
        "vehicle_id": 1,
        "start_date": contract_start,
        "monthly_fee": "R 250.00",
        "type": "Stolen Vehicle Recovery",
        "status": "Active",
        "expiry": (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
    }
        
    return {
        "user_id": user_id,
        "contracts": [contract],
        "services": ["Tracking", "Recovery", "App Access"],
        "service_level": "Standard",
        "retrieved_from": "contact_details_valid_state table"
    }

@tool
def get_bank_account_details(user_id: int) -> Dict[str, Any]:
    """
    Retrieve client's banking details.
    
    Args:
        user_id: The client's unique identifier
        
    Returns:
        Dictionary with banking information
    """
    # In a real implementation, this would call: ct.sp_debtors_account_info
    
    # Simulate bank details
    banks = ["FNB", "Standard Bank", "ABSA", "Nedbank", "Capitec"]
    bank_name = banks[user_id % len(banks)]
    
    return {
        "user_id": user_id,
        "bank_details": {
            "bank_name": bank_name,
            "account_number": f"{random.randint(1000, 9999)}******",
            "account_type": random.choice(["Cheque", "Savings"]),
            "branch_code": f"{random.randint(100, 999)}",
            "account_holder": "John Doe",
            "salary_date": f"{random.randint(25, 31)}"
        },
        "debit_order_frequency": "Monthly",
        "debit_day": random.randint(1, 28),
        "retrieved_from": "ct.sp_debtors_account_info"
    }

@tool
def get_payment_history(user_id: int) -> Dict[str, Any]:
    """
    Retrieve client's payment history including PTPs and debit order results.
    
    Args:
        user_id: The client's unique identifier
        
    Returns:
        Dictionary with payment history
    """
    # In a real implementation, this would access: ct.arrangement + ct.arrangement_pay tables
    
    # Generate simulated payment history
    last_three_months = []
    ptp_history = []
    
    for i in range(3):
        payment_date = (datetime.now() - timedelta(days=30*i)).strftime("%Y-%m-%d")
        
        # Random payment status
        status = random.choice(["Successful", "Failed", "Successful", "Reversed", "Successful"])
        
        last_three_months.append({
            "date": payment_date,
            "amount": f"R {random.randint(200, 500):.2f}",
            "type": "Debit Order",
            "status": status,
            "reference": f"PAY{random.randint(10000, 99999)}"
        })
    
    # Generate PTP history
    for i in range(2):
        ptp_date = (datetime.now() - timedelta(days=45*i)).strftime("%Y-%m-%d")
        fulfilled = random.choice([True, False])
        
        ptp_history.append({
            "date_created": (datetime.now() - timedelta(days=45*i+5)).strftime("%Y-%m-%d"),
            "promised_date": ptp_date,
            "amount": f"R {random.randint(500, 2000):.2f}",
            "fulfilled": fulfilled,
            "status": "Honored" if fulfilled else "Broken",
            "agent": "Agent" + str(random.randint(1, 10))
        })
        
    return {
        "user_id": user_id,
        "last_three_months": last_three_months,
        "ptp_history": ptp_history,
        "reversals_count": sum(1 for payment in last_three_months if payment["status"] == "Reversed"),
        "failed_count": sum(1 for payment in last_three_months if payment["status"] == "Failed"),
        "broken_ptp_count": sum(1 for ptp in ptp_history if not ptp["fulfilled"]),
        "retrieved_from": "ct.arrangement + ct.arrangement_pay tables"
    }

@tool
def get_mandate_status(user_id: int) -> Dict[str, Any]:
    """
    Retrieve client's DebiCheck mandate status.
    
    Args:
        user_id: The client's unique identifier
        
    Returns:
        Dictionary with mandate information
    """
    # In a real implementation, this would access: ct.client_debit_order_mandate table
    
    # Simulate mandate statuses
    statuses = ["Active", "Pending Authentication", "Expired", "None"]
    status = statuses[user_id % len(statuses)]
    
    mandate_info = {
        "user_id": user_id,
        "has_mandate": status != "None",
        "mandate_status": status,
        "retrieved_from": "ct.client_debit_order_mandate table"
    }
    
    if status != "None":
        mandate_info["mandate_details"] = {
            "created_date": (datetime.now() - timedelta(days=random.randint(30, 300))).strftime("%Y-%m-%d"),
            "expiry_date": (datetime.now() + timedelta(days=random.randint(30, 730))).strftime("%Y-%m-%d"),
            "maximum_amount": f"R {random.randint(500, 5000):.2f}",
            "bank_reference": f"DCM{random.randint(10000, 99999)}"
        }
        
    return mandate_info

@tool
def get_cancellation_fee(user_id: int, contract_id: str = None) -> Dict[str, Any]:
    """
    Retrieve cancellation fees for a client's contracts.
    
    Args:
        user_id: The client's unique identifier
        contract_id: Specific contract ID (optional, retrieves all if not specified)
        
    Returns:
        Dictionary with cancellation fees
    """
    # In a real implementation, this would access contract and cancellation fee tables
    
    # Generate simulated cancellation fee
    monthly_fee = 250.00
    
    # Calculate months remaining and cancellation fee
    today = datetime.now()
    expiry_date = datetime.strptime("2025-12-31", "%Y-%m-%d")
    months_remaining = (expiry_date.year - today.year) * 12 + expiry_date.month - today.month
    
    if months_remaining < 0:
        months_remaining = 0
        
    cancellation_fee = monthly_fee * min(months_remaining, 12)  # Cap at 12 months
    
    cancellation_details = [{
        "contract_id": "12345",
        "monthly_fee": f"R {monthly_fee:.2f}",
        "months_remaining": months_remaining,
        "cancellation_fee": f"R {cancellation_fee:.2f}"
    }]
    
    # Outstanding amount and total balance
    outstanding_amount = 1850.00
    total_balance = outstanding_amount + cancellation_fee
        
    return {
        "user_id": user_id,
        "cancellation_details": cancellation_details,
        "outstanding_amount": f"R {outstanding_amount:.2f}",
        "total_cancellation_fee": f"R {cancellation_fee:.2f}",
        "total_balance": f"R {total_balance:.2f}",
        "calculation_date": datetime.now().strftime("%Y-%m-%d"),
        "retrieved_from": "ct.contract, ct.cancellation_fee, ct.calc_outstanding"
    }

# -------------------------------------------------------------------------------------
# Action/Update tools
# -------------------------------------------------------------------------------------
@tool
def update_arrangement(user_id: int, amount: str, payment_date: str, 
                       payment_method: str, bank_details: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create or update a payment arrangement (PTP).
    
    Args:
        user_id: The client's unique identifier
        amount: Amount agreed for payment
        payment_date: Date scheduled for payment
        payment_method: Method of payment (DebiCheck, Immediate Debit, etc.)
        bank_details: Optional updated banking details
        
    Returns:
        Dictionary with arrangement details
    """
    # In a real implementation, this would call: bl.sp_update_arrangement
    
    # Generate arrangement details
    arrangement_id = f"ARR-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    
    arrangement = {
        "user_id": user_id,
        "arrangement_id": arrangement_id,
        "amount": amount,
        "payment_date": payment_date,
        "payment_method": payment_method,
        "status": "Scheduled",
        "created_by": "AI Agent",
        "created_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result": "success",
        "process_message": "Payment arrangement created successfully"
    }
    
    if bank_details:
        arrangement["bank_details_updated"] = True
    
    return arrangement

@tool
def update_bank_details(user_id: int, bank_name: str, account_number: str,
                        account_type: str, branch_code: str, account_holder: str,
                        salary_date: str) -> Dict[str, Any]:
    """
    Update client's banking details.
    
    Args:
        user_id: The client's unique identifier
        bank_name: Name of the bank
        account_number: Bank account number
        account_type: Type of account (Savings, Cheque)
        branch_code: Bank branch code
        account_holder: Name of account holder
        salary_date: Day of month when salary is received
        
    Returns:
        Dictionary with update status
    """
    # In a real implementation, this would update: ct.user_bank_account table
    
    return {
        "user_id": user_id,
        "update_status": "success",
        "bank_details": {
            "bank_name": bank_name,
            "account_number": f"{account_number[:4]}******",
            "account_type": account_type,
            "branch_code": branch_code,
            "account_holder": account_holder,
            "salary_date": salary_date
        },
        "updated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_by": "AI Agent",
        "message": "Bank details updated successfully"
    }

@tool
def send_payment_link(user_id: int, phone_number: str, amount: str) -> Dict[str, Any]:
    """
    Send an SMS payment link to client.
    
    Args:
        user_id: The client's unique identifier
        phone_number: Client's phone number
        amount: Amount to be paid
        
    Returns:
        Dictionary with SMS status
    """
    # In a real implementation, this would call: cc.url_payment_encode function
    
    # Generate payment link details
    link_id = f"PAY-{random.randint(100000, 999999)}"
    
    return {
        "user_id": user_id,
        "sms_sent": True,
        "phone_number": phone_number,
        "payment_link": f"https://pay.cartrack.co.za/{link_id}",
        "amount": amount,
        "link_expiry": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
        "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sent_by": "AI Agent",
        "message": f"Payment link for {amount} sent successfully to {phone_number}"
    }

@tool
def setup_debicheck(user_id: int, bank_name: str, account_number: str,
                   account_type: str, amount: str) -> Dict[str, Any]:
    """
    Set up DebiCheck authentication for client's payment.
    
    Args:
        user_id: The client's unique identifier
        bank_name: Name of the bank
        account_number: Bank account number
        account_type: Type of account (Savings, Cheque)
        amount: Amount to be debited
        
    Returns:
        Dictionary with DebiCheck setup status
    """
    # In a real implementation, this would set up DebiCheck authentication
    
    auth_reference = f"DC-{datetime.now().strftime('%y%m%d')}-{random.randint(1000, 9999)}"
    
    return {
        "user_id": user_id,
        "debicheck_reference": auth_reference,
        "status": "pending_approval",
        "bank": bank_name,
        "masked_account": f"{'*' * (len(account_number) - 4)}{account_number[-4:]}",
        "amount": amount,
        "setup_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "setup_by": "AI Agent",
        "estimated_completion": "15-30 minutes",
        "message": "DebiCheck authentication request sent to client's phone. They will need to approve through their banking app or USSD."
    }

@tool
def create_referral(user_id: int, referral_name: str, referral_phone: str,
                   referral_email: str = None) -> Dict[str, Any]:
    """
    Create a referral from client.
    
    Args:
        user_id: The client's unique identifier
        referral_name: Name of the person being referred
        referral_phone: Phone number of the referral
        referral_email: Email address of the referral (optional)
        
    Returns:
        Dictionary with referral status
    """
    # In a real implementation, this would update: ct.t_referral table
    
    referral_id = f"REF-{datetime.now().strftime('%y%m%d')}-{random.randint(1000, 9999)}"
    
    return {
        "user_id": user_id,
        "referral_id": referral_id,
        "referral_details": {
            "name": referral_name,
            "phone": referral_phone,
            "email": referral_email
        },
        "status": "Active",
        "created_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "created_by": "AI Agent",
        "reward_status": "Pending",
        "message": "Referral created successfully. Client will receive 2 months free subscription when referral signs up."
    }

@tool
def create_helpdesk_ticket(user_id: int, subject: str, details: str, 
                         ticket_type: str, priority: str = "Normal") -> Dict[str, Any]:
    """
    Create a helpdesk ticket for a client.
    
    Args:
        user_id: The client's unique identifier
        subject: Ticket subject
        details: Detailed description
        ticket_type: Type of ticket (Cancellation, Technical, Billing, etc.)
        priority: Ticket priority (Low, Normal, High, Critical)
        
    Returns:
        Dictionary with ticket details
    """
    # In a real implementation, this would update: ct.helpdesk table
    
    ticket_number = f"HD-{datetime.now().strftime('%y%m%d')}-{random.randint(1000, 9999)}"
    
    return {
        "user_id": user_id,
        "ticket_number": ticket_number,
        "subject": subject,
        "details": details,
        "type": ticket_type,
        "priority": priority,
        "status": "Open",
        "created_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "created_by": "AI Agent",
        "assigned_to": f"Department: {ticket_type}",
        "estimated_response": "24-48 hours",
        "message": f"Helpdesk ticket {ticket_number} created successfully."
    }

@tool
def add_client_note(user_id: int, note_text: str, note_type: str = "Call") -> Dict[str, Any]:
    """
    Add a note to client's account.
    
    Args:
        user_id: The client's unique identifier
        note_text: Content of the note
        note_type: Type of note (Call, Email, SMS, etc.)
        
    Returns:
        Dictionary with note details
    """
    # In a real implementation, this would update: ct.client_note table
    
    note_id = f"NOTE-{datetime.now().strftime('%y%m%d')}-{random.randint(1000, 9999)}"
    
    return {
        "user_id": user_id,
        "note_id": note_id,
        "note_text": note_text,
        "note_type": note_type,
        "created_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "created_by": "AI Agent",
        "message": "Note added successfully to client's account."
    }

@tool
def update_client_details(user_id: int, contact_number: str = None, email: str = None, 
                        address: Dict[str, str] = None, next_of_kin: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Update client's contact details.
    
    Args:
        user_id: The client's unique identifier
        contact_number: Updated contact number
        email: Updated email address
        address: Updated address information
        next_of_kin: Updated next of kin information
        
    Returns:
        Dictionary with update status
    """
    # In a real implementation, this would update: ct.individual, ct.individual_extra tables
    
    updates = {}
    if contact_number:
        updates["contact_number"] = contact_number
    if email:
        updates["email"] = email
    if address:
        updates["address"] = address
    if next_of_kin:
        updates["next_of_kin"] = next_of_kin
        
    update_count = len(updates)
    
    return {
        "user_id": user_id,
        "updates_successful": True,
        "fields_updated": update_count,
        "updated_fields": list(updates.keys()),
        "updated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_by": "AI Agent",
        "message": f"Successfully updated {update_count} client detail fields."
    }

@tool
def save_call_disposition(user_id: int, disposition_code: str, notes: str = None) -> Dict[str, Any]:
    """
    Save the call disposition code and close the case.
    
    Args:
        user_id: The client's unique identifier
        disposition_code: Code indicating call outcome
        notes: Additional notes about disposition
        
    Returns:
        Dictionary with disposition details
    """
    # In a real implementation, this would update: ct.t_debtor_disposition table
    
    disposition_id = f"DISP-{datetime.now().strftime('%y%m%d')}-{random.randint(1000, 9999)}"
    
    return {
        "user_id": user_id,
        "disposition_id": disposition_id,
        "disposition_code": disposition_code,
        "notes": notes,
        "created_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "created_by": "AI Agent",
        "case_closed": True,
        "message": f"Call disposition saved and case closed successfully."
    }

@tool
def escalate_to_department(user_id: int, department: str, reason: str, 
                         priority: str = "Normal") -> Dict[str, Any]:
    """
    Escalate a client case to a specific department.
    
    Args:
        user_id: The client's unique identifier
        department: Department to escalate to (Finance, Technical, Management)
        reason: Reason for escalation
        priority: Escalation priority (Low, Normal, High, Critical)
        
    Returns:
        Dictionary with escalation details
    """
    # This would create a specialized ticket/case in a real implementation
    
    escalation_id = f"ESC-{datetime.now().strftime('%y%m%d')}-{random.randint(1000, 9999)}"
    
    return {
        "user_id": user_id,
        "escalation_id": escalation_id,
        "department": department,
        "reason": reason,
        "priority": priority,
        "status": "Pending",
        "created_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "created_by": "AI Agent",
        "estimated_response": "24 hours" if priority in ["High", "Critical"] else "48 hours",
        "message": f"Case escalated to {department} department successfully."
    }

# -------------------------------------------------------------------------------------
# Call flow process tools
# -------------------------------------------------------------------------------------
@tool
def check_immediate_debit_time() -> Dict[str, Any]:
    """
    Check if an immediate debit can be processed today based on the current time.
    Returns information about processing time and debit date.
    """
    current_hour = datetime.now().hour
    before_2pm = current_hour < 14
    
    if before_2pm:
        return {
            "before_2pm": True,
            "debit_date": datetime.now().strftime("%Y-%m-%d"),
            "debit_time": "today",
            "message": "Since it's before 2PM, the debit will be processed today."
        }
    else:
        next_day = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        return {
            "before_2pm": False,
            "debit_date": next_day,
            "debit_time": "tomorrow",
            "message": "Since it's after 2PM, the debit will be processed tomorrow."
        }

@tool
def check_client_payment_history(user_id: int) -> Dict[str, Any]:
    """
    Check the client's payment history for previous failed payments or reversals.
    Returns information about payment history and suggested actions.
    
    Args:
        user_id: The client's unique identifier
    
    Returns:
        Dictionary with payment history analysis and recommendations
    """
    # Call our database function to get actual payment history
    payment_history = get_payment_history(user_id)
    
    # Extract relevant information
    failed_ptps = payment_history.get("broken_ptp_count", 0) > 0
    reversals = payment_history.get("reversals_count", 0) > 0
    
    # Determine the suggestion based on payment history
    suggestion = "proceed with normal debit order"
    if failed_ptps and reversals:
        suggestion = "explain impact of failed payments and reversals, request confirmation, and new bank details"
    elif failed_ptps:
        suggestion = "explain impact of failed payments and request new bank details"
    elif reversals:
        suggestion = "explain impact of reversals and request confirmation"
    
    return {
        "failed_ptps": failed_ptps,
        "reversals": reversals,
        "bank_details_changed": False,  # Would check in real implementation
        "suggestion": suggestion,
        "history_summary": {
            "failed_payments": payment_history.get("failed_count", 0),
            "reversals": payment_history.get("reversals_count", 0),
            "broken_ptps": payment_history.get("broken_ptp_count", 0)
        }
    }

@tool
def process_payment_portal_sms(user_id: int, phone_number: str = None, amount: str = None) -> Dict[str, Any]:
    """
    Send a payment portal SMS to the client's phone number.
    
    Args:
        user_id: The client's unique identifier
        phone_number: Client's phone number to send the SMS to (optional)
        amount: Amount to be paid (optional)
        
    Returns:
        Dictionary with SMS status and details
    """
    # Get phone number and amount from client info if not provided
    if not phone_number:
        phone_number = "+27123456789"  # Default for testing
        
    if not amount:
        amount = "R 1850.00"  # Default for testing
    
    # Ensure amount has correct format
    if not amount.startswith("R "):
        amount = f"R {amount.replace('R', '').strip()}"
    
    # Call send_payment_link to send the SMS
    return send_payment_link(user_id, phone_number, amount)

@tool
def log_cancellation_ticket(user_id: int, reason: str = "Client request") -> Dict[str, Any]:
    """
    Log a cancellation ticket in the helpdesk system.
    
    Args:
        user_id: The client's unique identifier
        reason: Reason for cancellation
        
    Returns:
        Dictionary with ticket details
    """
    # Call our database function to create the ticket
    return create_helpdesk_ticket(
        user_id=user_id,
        subject="Cancellation Request",
        details=f"Reason: {reason}. Client has requested to cancel their contract.",
        ticket_type="Cancellation",
        priority="High"
    )

@tool
def check_for_referrals(response_text: str) -> Dict[str, Any]:
    """
    Analyze client response to determine if they have referrals.
    
    Args:
        response_text: Client's response to referral question
        
    Returns:
        Dictionary with referral analysis
    """
    has_referrals = False
    content = response_text.lower()
    
    # Check for referral indications
    if any(term in content for term in ["yes", "know someone", "friend", "colleague", "family"]):
        if not any(term in content for term in ["no", "don't know", "not really", "nobody"]):
            has_referrals = True
    
    if has_referrals:
        return {
            "has_referrals": True,
            "action": "capture referral details",
            "message": "Client indicated they have referrals. Collect their contact details."
        }
    else:
        return {
            "has_referrals": False,
            "action": "proceed to further assistance",
            "message": "Client did not indicate any referrals."
        }

@tool
def process_debicheck_setup(user_id: int, bank_name: str, account_number: str, account_type: str, amount: str) -> Dict[str, Any]:
    """
    Process DebiCheck setup for client payment.
    
    Args:
        user_id: The client's unique identifier
        bank_name: Client's bank name
        account_number: Account number for debit
        account_type: Type of account (Savings, Cheque)
        amount: Amount including R10 fee
        
    Returns:
        Dictionary with DebiCheck setup details
    """
    # Call our database function to set up DebiCheck
    return setup_debicheck(
        user_id=user_id,
        bank_name=bank_name,
        account_number=account_number,
        account_type=account_type,
        amount=amount
    )

@tool
def save_client_referral(user_id: int, referral_name: str, referral_phone: str, referral_email: str = None) -> Dict[str, Any]:
    """
    Save a client referral.
    
    Args:
        user_id: The client's unique identifier
        referral_name: Name of the person being referred
        referral_phone: Phone number of the referral
        referral_email: Email of the referral (optional)
        
    Returns:
        Dictionary with referral details
    """
    # Call our database function to create the referral
    return create_referral(
        user_id=user_id,
        referral_name=referral_name,
        referral_phone=referral_phone,
        referral_email=referral_email
    )

@tool
def process_call_disposition(user_id: int, disposition_code: str, call_notes: str) -> Dict[str, Any]:
    """
    Process the call disposition and add detailed notes.
    
    Args:
        user_id: The client's unique identifier
        disposition_code: Code indicating call outcome
        call_notes: Detailed notes about the call
        
    Returns:
        Dictionary with disposition details
    """
    # First add the client note
    note_result = add_client_note(
        user_id=user_id,
        note_text=call_notes,
        note_type="Call"
    )
    
    # Then save the disposition
    disposition_result = save_call_disposition(
        user_id=user_id,
        disposition_code=disposition_code,
        notes=f"Call notes added with ID: {note_result.get('note_id')}"
    )
    
    return {
        "note_id": note_result.get("note_id"),
        "disposition_id": disposition_result.get("disposition_id"),
        "disposition_code": disposition_code,
        "notes_saved": True,
        "disposition_saved": True,
        "case_closed": disposition_result.get("case_closed", False),
        "message": "Call documentation completed successfully."
    }

@tool
def process_escalation(user_id: int, department: str, reason: str, priority: str = "Normal") -> Dict[str, Any]:
    """
    Process an escalation to another department.
    
    Args:
        user_id: The client's unique identifier
        department: Department to escalate to
        reason: Reason for escalation
        priority: Priority level
        
    Returns:
        Dictionary with escalation details
    """
    # Call our database function to escalate
    return escalate_to_department(
        user_id=user_id,
        department=department,
        reason=reason,
        priority=priority
    )