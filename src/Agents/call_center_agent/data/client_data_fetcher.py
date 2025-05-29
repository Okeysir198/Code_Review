# src/Agents/call_center_agent/data/client_data_fetcher.py
"""
Simple client data fetching and outstanding amount calculation
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Import database tools
from src.Database.CartrackSQLDatabase import (
    get_client_profile,
    get_client_account_overview,
    get_client_account_aging,
    get_client_banking_details,
    get_client_subscription_amount,
    get_client_payment_history,
    get_client_failed_payments,
    get_client_last_successful_payment,
    get_client_contracts,
    get_client_billing_analysis,
    get_client_debit_mandates
)

logger = logging.getLogger(__name__)

# Simple cache
_cache = {}
_cache_duration = timedelta(hours=1)

def get_client_data(user_id: str, force_reload: bool = False) -> Dict[str, Any]:
    """Get client data with simple caching."""
    cache_key = user_id
    now = datetime.now()
    
    # Check cache
    if not force_reload and cache_key in _cache:
        cached_entry = _cache[cache_key]
        if now - cached_entry["timestamp"] < _cache_duration:
            logger.info(f"Using cached data for user_id: {user_id}")
            return cached_entry["data"]
    
    # Fetch fresh data
    logger.info(f"Fetching fresh data for user_id: {user_id}")
    try:
        data = _fetch_client_data(user_id)
        _cache[cache_key] = {"data": data, "timestamp": now}
        return data
    except Exception as e:
        logger.error(f"Error fetching client data for {user_id}: {e}")
        return _get_fallback_data(user_id)

def _fetch_client_data(user_id: str) -> Dict[str, Any]:
    """Fetch data from database."""
    try:
        # Load core client information
        profile = get_client_profile.invoke(user_id)
        if not profile:
            raise ValueError(f"Client profile not found for user_id: {user_id}")
        
        # Load account and financial data
        account_overview = get_client_account_overview.invoke(user_id)
        account_aging = get_client_account_aging.invoke(user_id)
        banking_details = get_client_banking_details.invoke(user_id)
        subscription_data = get_client_subscription_amount.invoke(user_id)
        
        # Consolidate data
        client_data = {
            "user_id": user_id,
            "profile": profile,
            "account_overview": account_overview,
            "account_aging": account_aging[0] if account_aging else {},
            "banking_details": banking_details[0] if banking_details else {},
            "subscription": subscription_data,
            "loaded_at": datetime.now()
        }
        
        logger.info(f"Successfully loaded data for user_id: {user_id}")
        return client_data
        
    except Exception as e:
        logger.error(f"Error loading client data for {user_id}: {str(e)}")
        raise

def _get_fallback_data(user_id: str) -> Dict[str, Any]:
    """Fallback data when database fails."""
    return {
        "user_id": user_id,
        "profile": {
            "client_info": {
                "client_full_name": "Client",
                "first_name": "Client",
                "title": "Mr/Ms"
            }
        },
        "account_overview": {"account_status": "Overdue"},
        "account_aging": {"xbalance": "0.00", "x0": "0.00"},
        "banking_details": {},
        "subscription": {"subscription_amount": "199.00"},
        "loaded_at": datetime.now(),
        "fallback_used": True
    }

def calculate_outstanding_amount(account_aging: Dict[str, Any]) -> float:
    """
    Calculate outstanding amount = total balance - current (non-overdue) balance.
    This is the OVERDUE amount the client needs to pay.
    """
    try:
        total_balance = float(account_aging.get("xbalance", 0))
        current_balance = float(account_aging.get("x0", 0))
        outstanding = total_balance - current_balance
        return max(outstanding, 0.0)  # Never negative
    except (ValueError, TypeError):
        return 0.0

def format_currency(amount: float) -> str:
    """Format amount as currency."""
    return f"R {amount:.2f}"

def get_safe_value(data: Dict[str, Any], path: str, default: Any = "") -> Any:
    """Safely extract nested values with dot notation."""
    try:
        keys = path.split('.')
        value = data
        for key in keys:
            value = value[key] if isinstance(value, dict) else getattr(value, key, None)
        return value if value is not None else default
    except (KeyError, TypeError, AttributeError):
        return default

def clear_cache(user_id: Optional[str] = None):
    """Clear cached client data."""
    if user_id:
        _cache.pop(user_id, None)
        logger.info(f"Cleared cache for user {user_id}")
    else:
        _cache.clear()
        logger.info("Cleared all cache")
