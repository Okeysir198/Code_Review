# src/Agents/call_center_agent/data/client_data_fetcher.py
"""
Lean client data fetching with concurrent loading and minimal caching
"""
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import database tools
from src.Database.CartrackSQLDatabase import (
    get_client_profile,
    get_client_account_overview,
    get_client_account_aging,
    get_client_banking_details,
    get_client_subscription_amount,
    get_client_payment_history,
    get_client_account_statement,
    get_client_debit_mandates
)

logger = logging.getLogger(__name__)

# Minimal cache with shorter duration
_cache = {}
_cache_duration = timedelta(minutes=15)

# =====================================================
# Main Functions
# =====================================================

def get_client_data(user_id: str, force_reload: bool = False, use_connection_pool: bool = True) -> Dict[str, Any]:
    """
    Get client data with concurrent fetching and minimal caching.
    
    Args:
        user_id: Client's unique identifier
        force_reload: Skip cache and fetch fresh data
        use_connection_pool: Use connection pool for better performance (default: True)
    
    Returns:
        Dict containing all client data
    """
    
    # Check cache first (if not forcing reload)
    if not force_reload:
        cached_data = _get_cached_data(user_id)
        if cached_data:
            return cached_data
    
    # Fetch fresh data concurrently
    if use_connection_pool:
        return _fetch_with_connection_pool(user_id)
    else:
        return _fetch_concurrent_data(user_id)

def get_client_data_fast(user_id: str, force_reload: bool = False) -> Dict[str, Any]:
    """
    Fastest client data fetching using connection pool.
    Alias for get_client_data with connection pool enabled.
    """
    return get_client_data(user_id, force_reload, use_connection_pool=True)

# =====================================================
# Core Data Fetching Functions
# =====================================================

def _fetch_with_connection_pool(user_id: str) -> Dict[str, Any]:
    """
    Fetch client data using connection pool for maximum performance.
    Each thread gets its own database connection from the pool.
    """
    start_time = time.time()
    
    # Define tasks using the original tool.invoke pattern
    fetch_tasks = {
        'profile': lambda: get_client_profile.invoke(user_id),
        'account_overview': lambda: get_client_account_overview.invoke(user_id),
        'account_aging': lambda: get_client_account_aging.invoke(user_id),
        'banking_details': lambda: get_client_banking_details.invoke(user_id),
        'subscription': lambda: get_client_subscription_amount.invoke(user_id),
        'payment_history': lambda: get_client_payment_history.invoke(user_id),
    }
    
    results = {}
    failed_tasks = []
    
    # Execute with connection pool - each thread gets its own connection
    with ThreadPoolExecutor(max_workers=len(fetch_tasks)) as executor:
        future_to_name = {
            executor.submit(task): name 
            for name, task in fetch_tasks.items()
        }
        
        for future in as_completed(future_to_name):
            task_name = future_to_name[future]
            try:
                result = future.result()
                results[task_name] = result
                logger.debug(f"✓ {task_name} completed")
            except Exception as e:
                failed_tasks.append(task_name)
                results[task_name] = None
                logger.warning(f"✗ {task_name} failed for user {user_id}: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"Connection pool fetch completed in {elapsed:.3f}s for user {user_id}")
    
    return _build_client_data_response(user_id, results, failed_tasks, elapsed, "connection_pool")

def _fetch_concurrent_data(user_id: str) -> Dict[str, Any]:
    """
    Fetch data using original concurrent method (for fallback/comparison).
    """
    start_time = time.time()
    
    # Define data fetching tasks
    fetch_tasks = {
        'profile': lambda: get_client_profile.invoke(user_id),
        'account_overview': lambda: get_client_account_overview.invoke(user_id),
        'account_aging': lambda: get_client_account_aging.invoke(user_id),
        'banking_details': lambda: get_client_banking_details.invoke(user_id),
        'subscription': lambda: get_client_subscription_amount.invoke(user_id),
        'payment_history': lambda: get_client_payment_history.invoke(user_id),
    }
    
    results = {}
    failed_tasks = []
    
    # Execute all tasks concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_name = {
            executor.submit(task): name 
            for name, task in fetch_tasks.items()
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_name):
            task_name = future_to_name[future]
            try:
                result = future.result(timeout=10)  # 10 second timeout per task
                results[task_name] = result
                logger.debug(f"✓ {task_name} loaded for user {user_id}")
            except Exception as e:
                failed_tasks.append(task_name)
                results[task_name] = None
                logger.warning(f"✗ {task_name} failed for user {user_id}: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"Concurrent fetch completed in {elapsed:.3f}s for user {user_id}")
    
    return _build_client_data_response(user_id, results, failed_tasks, elapsed, "concurrent")

# =====================================================
# Helper Functions
# =====================================================

def _build_client_data_response(user_id: str, results: Dict, failed_tasks: list, 
                               elapsed: float, fetch_method: str) -> Dict[str, Any]:
    """
    Build the standardized client data response structure.
    """
    # Validate critical data
    if not results.get('profile'):
        raise ValueError(f"Critical data missing: client profile not found for user_id: {user_id}")
    
    # Build consolidated data structure
    client_data = {
        "user_id": user_id,
        "profile": results['profile'],
        "account_overview": results['account_overview'],
        "account_aging": _extract_first_item(results['account_aging']),
        "banking_details": _extract_first_item(results['banking_details']),
        "subscription": results['subscription'] or {},
        "payment_history": results['payment_history'], 
        "loaded_at": datetime.now(),
        "failed_tasks": failed_tasks if failed_tasks else None,
        "fetch_time_seconds": round(elapsed, 3),
        "fetch_method": fetch_method
    }
    
    # Cache the result
    _cache[user_id] = {
        "data": client_data,
        "timestamp": datetime.now()
    }
    
    logger.info(f"Data loaded for user {user_id} - {len(failed_tasks)} failures - Method: {fetch_method}")
    return client_data

def _get_cached_data(user_id: str) -> Optional[Dict[str, Any]]:
    """Check cache for valid data."""
    if user_id not in _cache:
        return None
        
    cached_entry = _cache[user_id]
    if datetime.now() - cached_entry["timestamp"] < _cache_duration:
        logger.debug(f"Cache hit for user_id: {user_id}")
        return cached_entry["data"]
    
    # Remove stale cache entry
    del _cache[user_id]
    return None

def _extract_first_item(data_list):
    """Extract first item from list or return empty dict."""
    if isinstance(data_list, list) and len(data_list) > 0:
        return data_list[0]
    return {}

# =====================================================
# Utility Functions
# =====================================================

def calculate_outstanding_amount(account_aging: Dict[str, Any]) -> float:
    """Calculate overdue amount (total - current)."""
    try:
        total = float(account_aging.get("xbalance", 0))
        current = float(account_aging.get("x0", 0))
        return max(total - current, 0.0)
    except (ValueError, TypeError):
        return 0.0

def format_currency(amount: float) -> str:
    """Format amount as currency string."""
    return f"R {amount:.2f}"

def get_safe_value(data: Dict[str, Any], path: str, default: Any = "") -> Any:
    """Extract nested values using dot notation."""
    try:
        current = data
        for key in path.split('.'):
            current = current[key] if isinstance(current, dict) else getattr(current, key)
        return current if current is not None else default
    except (KeyError, TypeError, AttributeError):
        return default

# =====================================================
# Cache Management
# =====================================================

def clear_cache(user_id: Optional[str] = None) -> None:
    """Clear cached data."""
    if user_id:
        _cache.pop(user_id, None)
        logger.info(f"Cache cleared for user {user_id}")
    else:
        _cache.clear()
        logger.info("All cache cleared")

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for monitoring."""
    now = datetime.now()
    valid_entries = sum(
        1 for entry in _cache.values() 
        if now - entry["timestamp"] < _cache_duration
    )
    
    return {
        "total_entries": len(_cache),
        "valid_entries": valid_entries,
        "cache_hit_ratio": valid_entries / max(len(_cache), 1),
        "cache_duration_minutes": _cache_duration.total_seconds() / 60
    }
