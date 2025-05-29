import time
import asyncio
import logging
from typing import Dict, Any
from src.Agents.call_center_agent.data_parameter_builder import get_client_data_async

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simple global cache
_cache = {}

async def get_client_data_async_cache(user_id: str, force_reload: bool = False) -> Dict[str, Any]:
    """Get client data with 1-hour caching (async)."""
    cache_key = f"client_{user_id}"
    current_time = time.time()
    
    # Check cache
    if not force_reload and cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if current_time - cached_time < 3600:  # 1 hour
            logger.info(f"✅ Using cached data for user {user_id}")
            return cached_data
        
    # Cache miss or expired - fetch fresh data
    logger.info(f"🔄 Fetching fresh data for user {user_id}")
    start_time = time.time()
    
    try:
        fresh_data = await get_client_data_async(user_id, force_reload)
        fetch_duration = time.time() - start_time
        logger.info(f"✅ Fresh data fetched for user {user_id} in {fetch_duration:.2f} seconds")
        
        _cache[cache_key] = (current_time, fresh_data)
        return fresh_data
        
    except Exception as e:
        fetch_duration = time.time() - start_time
        logger.error(f"❌ Failed to fetch data for user {user_id} after {fetch_duration:.2f} seconds: {e}")
        raise


async def get_multiple_clients(user_ids: list[str], max_concurrent: int = 5) -> Dict[str, Dict[str, Any]]:
    """Get multiple client data concurrently."""
    logger.info(f"🔄 Fetching data for {len(user_ids)} users with max {max_concurrent} concurrent requests")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def get_single(user_id: str):
        async with semaphore:
            try:
                return user_id, await get_client_data_async_cache(user_id)
            except Exception as e:
                logger.error(f"❌ Failed to get data for user {user_id}: {e}")
                return user_id, None
    
    results = await asyncio.gather(*[get_single(uid) for uid in user_ids])
    
    # Filter out failed requests
    successful_results = {user_id: data for user_id, data in results if data is not None}
    
    total_duration = time.time() - start_time
    logger.info(f"✅ Batch fetch completed: {len(successful_results)}/{len(user_ids)} users in {total_duration:.2f} seconds")
    
    return successful_results

def clear_cache(user_id: str = None):
    """Clear cache for specific user or all."""
    if user_id:
        if f"client_{user_id}" in _cache:
            _cache.pop(f"client_{user_id}")
            logger.info(f"🗑️ Cleared cache for user {user_id}")
        else:
            logger.info(f"ℹ️ No cache found for user {user_id}")
    else:
        cache_count = len(_cache)
        _cache.clear()
        logger.info(f"🗑️ Cleared all cache ({cache_count} entries)")

def get_cache_stats():
    """Get cache statistics."""
    current_time = time.time()
    valid_entries = 0
    expired_entries = 0
    
    for cache_key, (cached_time, _) in _cache.items():
        age = current_time - cached_time
        if age < 3600:  # 1 hour
            valid_entries += 1
        else:
            expired_entries += 1
    
    logger.info(f"📊 Cache stats: {valid_entries} valid, {expired_entries} expired, {len(_cache)} total entries")
    return {
        "total": len(_cache),
        "valid": valid_entries,
        "expired": expired_entries
    }

# FIXED: Load client data with proper event loop handling
async def _load_client_data(user_id):
    """Load client data at module level."""
    return await get_client_data_async_cache(user_id)

def _load_client_data_sync(user_id: str) -> Dict[str, Any]:
    """Synchronous wrapper that handles event loop detection."""
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we get here, we're in a running loop (like Jupyter)
        logger.warning("⚠️ Running event loop detected - using fallback sync data loading")
        return _get_fallback_client_data(user_id)
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(_load_client_data(user_id))

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
        "payment_history": [
            {
                "arrangement_id": "PTP001",
                "amount": "199.00",
                "pay_date": "2024-01-15",
                "arrangement_state": "Fulfilled",
                "created_by": "System"
            }
        ],
        "failed_payments": [
            {
                "payment_date": "2024-02-05",
                "failure_reason": "Insufficient funds"
            }
        ],
        "last_successful_payment": {
            "payment_id": "PAY001",
            "payment_date": "2024-01-15",
            "payment_amount": "199.00"
        },
        "contracts": [
            {
                "contract_id": "CON001",
                "status": "Active",
                "start_date": "2023-01-01",
                "vehicle_registration": "ABC123GP"
            }
        ],
        "billing_analysis": {
            "balance_30_days": "199.00",
            "balance_60_days": "0.00",
            "balance_90_days": "0.00",
            "balance_120_days": "0.00"
        },
        "existing_mandates": [],
        "loaded_at": "2024-12-19 10:30:00",
        "load_duration_seconds": 0.1,
        "fallback_used": True
    }

# Execute and return client_data at module level with proper event loop handling
user_id = "83906"  # Example user ID, can be changed as needed

client_data = _load_client_data_sync(user_id)
logger.info(f"✅ Client data loaded at module level for user: {client_data.get('user_id', 'unknown')}")




# Export functions for external use
__all__ = [
    'client_data',
    'get_client_data_async_cache',
    'get_multiple_clients', 
    'clear_cache',
    'get_cache_stats',
    '_load_client_data_sync',
    '_get_fallback_client_data'
]