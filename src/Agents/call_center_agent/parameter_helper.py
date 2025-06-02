# src/Agents/call_center_agent/parameter_helper.py
"""
Lean parameter preparation for consistent script and prompt formatting
"""
import logging
from datetime import datetime
from typing import Dict, Any
from src.Agents.call_center_agent.call_scripts import ScriptManager
from app_config import CONFIG

logger = logging.getLogger(__name__)

def prepare_parameters(
    client_data: Dict[str, Any], 
    state: Dict[str, Any], 
    agent_name: str = "AI Agent"
) -> Dict[str, str]:
    """
    Prepare all parameters needed for script and prompt formatting.
    
    Args:
        client_data: Client data from data fetcher
        state: Current call state
        agent_name: Agent name
        
    Returns:
        Dict with all formatted parameters
    """
    params = {}
    
    # === CORE IDENTIFIERS ===
    params["user_id"] = _get_safe_value(client_data, "profile.user_id", "")
    params["agent_name"] = agent_name
    
    # === CLIENT INFO ===
    client_info = client_data.get("profile", {}).get("client_info", {})
    params["client_title"] = client_info.get("title", "Mr/Ms")
    params["client_full_name"] = client_info.get("client_full_name", "Valued Client") 
    params["client_name"] = client_info.get("first_name", params["client_full_name"])
    params["salutation"] = f"Good day, {params['client_title']}"
    
    # === FINANCIAL INFO ===
    account_aging = client_data.get("account_aging", {})
    account_overview = client_data.get("account_overview", {})
    
    outstanding_float = _calculate_outstanding_amount(account_aging)
    params["outstanding_amount"] = _format_currency(outstanding_float)
    params["amount_with_fee"] = _format_currency(outstanding_float + 10)
    params["total_balance"] = _format_currency(account_aging.get("xbalance", 0))
    params["cancellation_fee"] = _format_currency(account_overview.get("cancellation_fee", 0))
    
    subscription = client_data.get("subscription", {})
    banking_details = client_data.get("banking_details", {})
    params["subscription_amount"] = _format_currency(subscription.get("subscription_amount", 199))
    params["subscription_date"] = f"{banking_details.get('debit_date')} of each month"
    
    params['banking_details'] = banking_details
    # === ACCOUNT STATUS ===
    params["account_status"] = account_overview.get("account_status", "Overdue")
    
    # === PAYMENT HISTORY (for PTP/Short paid scenarios) ===
    payment_history = client_data.get("payment_history", [])
    if payment_history:
        last_payment = payment_history[-1]
        params["paid_amount"] = _format_currency(last_payment.get("amount_paid", 0))
        params["payment_date"] = last_payment.get("payment_date", "recent date")
        params["agreed_amount"] = _format_currency(last_payment.get("agreed_amount", 0))
        params["shortfall_amount"] = _format_currency(last_payment.get("shortfall", 0))
    else:
        params.update({
            "paid_amount": "R 0.00", "payment_date": "N/A",
            "agreed_amount": "R 0.00", "shortfall_amount": "R 0.00"
        })
    
    # === AGING CONTEXT ===
    script_type = ScriptManager.determine_script_type_from_aging(account_aging, client_data)
    aging_context = ScriptManager.get_aging_context(script_type)
    
    params["aging_category"] = aging_context['category']
    params["urgency_level"] = aging_context['urgency']
    
    # === DISCOUNT INFO (for pre-legal scenarios) ===
    params["discount_percentage"] = state.get("discount_percentage", "50")
    base_amount = outstanding_float
    params["discounted_amount"] = _format_currency(base_amount * 0.6)  # 40% discount
    params["discounted_amount_50"] = _format_currency(base_amount * 0.5)  # 50% discount
    params["campaign_end_date"] = state.get("campaign_end_date", "month-end")
    params["campaign_first_date"] = state.get("campaign_first_date", "next week")
    
    # === VERIFICATION INFO ===
    params["field_to_verify"] = state.get("field_to_verify", "ID number")
    
    # === CONTACT INFO ===
    params["current_mobile"] = _get_safe_value(client_data, "profile.client_info.contact.mobile", "")
    params["current_email"] = _get_safe_value(client_data, "profile.client_info.email_address", "")
    
    # === VERIFICATION CONFIG INFO ===
    params["max_name_verification_attempts"] = CONFIG.get("verification", {}).get("max_name_verification_attempts", 5)
    params["max_details_verification_attempts"] = CONFIG.get("verification", {}).get("max_details_verification_attempts", 5)
    
    # === STATE-SPECIFIC INFO ===
    params["current_step"] = state.get("current_step", "Unknown Step")
    params["details_verification_status"] = state.get("details_verification_status", "Pending")
    params["name_verification_status"] = state.get("name_verification_status", "Pending")
    params["name_verification_attempts"] = state.get("name_verification_attempts", 0)
    params["details_verification_attempts"] = state.get("details_verification_attempts", 0)
    params["matched_fields"] = state.get("matched_fields", [])
    params["field_to_verify"] = state.get("field_to_verify", "ID number")
    params["outcome_summary"] = state.get("outcome_summary", "We appreciate your time")
    params["ticket_number"] = state.get("ticket_number", "")
    params["department"] = state.get("department", "Supervisor")
    params["response_time"] = state.get("response_time", "24-48 hours")
    params["return_to_step"] = state.get("return_to_step", "")
    params["redirect_message"] = state.get("redirect_message", "")
    params["last_client_question"] = state.get("last_client_question", "")
    params["payment_url"] = state.get("payment_url", "")
    params["call_outcome"] = state.get("call_outcome", "incomplete")
    params["should_offer_referrals"] = state.get("should_offer_referrals", False)
    

    params["current_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return params

def _get_safe_value(data: Dict[str, Any], path: str, default: str = "") -> str:
    """Extract nested value using dot notation"""
    try:
        current = data
        for key in path.split('.'):
            current = current[key]
        return str(current) if current is not None else default
    except (KeyError, TypeError):
        return default

def _calculate_outstanding_amount(account_aging: Dict[str, Any]) -> float:
    """Calculate overdue amount (total - current)"""
    try:
        total = float(account_aging.get("xbalance", 0))
        current = float(account_aging.get("x0", 0))
        return max(total - current, 0.0)
    except (ValueError, TypeError):
        return 0.0

def _format_currency(amount: Any) -> str:
    """Format amount as currency string"""
    try:
        return f"R {float(amount):.2f}"
    except (ValueError, TypeError):
        return "R 0.00"