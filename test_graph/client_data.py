import time
import asyncio
import logging
from typing import Dict, Any

# Use the new simplified data fetcher
from src.Agents.call_center_agent.data.client_data_fetcher import get_client_data, clear_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _load_client_data_sync(user_id: str) -> Dict[str, Any]:
    """Synchronous wrapper that handles event loop detection."""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we get here, we're in a running loop (like Jupyter)
        logger.warning("⚠️ Running event loop detected - using fallback sync data loading")
        return _get_fallback_client_data(user_id)
    except RuntimeError:
        # No running loop, safe to use the data fetcher
        return get_client_data(user_id)

def _get_fallback_client_data(user_id: str) -> Dict[str, Any]:
    """Fallback client data when async loading fails."""
    logger.info(f"🔄 Using fallback data for user {user_id}")
    
    return {
        "user_id": user_id,
        "profile": {
            "client_info": {
                "client_full_name": "John Smith",
                "first_name": "John",
                "title": "Mr",
                "id_number": "8001010001088",
                "email_address": "john.smith@example.com",
                "contact": {
                    "mobile": "0821234567"
                }
            },
            "user_name": "jsmith123",
            "vehicles": [
                {
                    "registration": "ABC123GP",
                    "make": "Toyota",
                    "model": "Camry",
                    "color": "White",
                    "chassis_number": "1234567890",
                    "model_year": "2020",
                    "contract_status": "Active"
                }
            ]
        },
        "account_overview": {
            "account_status": "Overdue"
        },
        "account_aging": {
            "x0": "0.00",       # Current
            "x30": "199.00",    # 1-30 days overdue
            "x60": "0.00",      # 31-60 days overdue
            "x90": "0.00",      # 61-90 days overdue
            "x120": "0.00",     # 91+ days overdue
            "xbalance": "199.00" # Total balance
        },
        "banking_details": {
            "account_number": "1234567890",
            "bank_name": "Standard Bank",
            "account_type": "Cheque"
        },
        "subscription": {
            "subscription_amount": "199.00"
        },
        "loaded_at": "2024-12-19 10:30:00",
        "fallback_used": True
    }

# Execute and return client_data at module level
user_id = "83906"  # Example user ID, can be changed as needed

client_data = _load_client_data_sync(user_id)
logger.info(f"✅ Client data loaded at module level for user: {client_data.get('user_id', 'unknown')}")

# Export functions for external use
__all__ = [
    'client_data',
    'get_client_data',
    'clear_cache',
    '_load_client_data_sync',
    '_get_fallback_client_data'
]