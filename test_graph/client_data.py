import time
from typing import Dict, Any, Optional

# Simple global cache
_cache = {}

def get_cached_client_data(user_id: str) -> Dict[str, Any]:
    """Get client data with 1-hour caching"""
    cache_key = f"client_{user_id}"
    current_time = time.time()
    
    # Check if data exists and is not expired (3600 seconds = 1 hour)
    if cache_key in _cache:
        cached_time, cached_data = _cache[cache_key]
        if current_time - cached_time < 3600:  # 1 hour = 3600 seconds
            print(f"✅ Using cached data for user {user_id}")
            return cached_data
    
    # Cache miss or expired - fetch fresh data
    print(f"🔄 Fetching fresh data for user {user_id}")
    from src.Agents.call_center_agent.data_parameter_builder import get_client_data
    
    fresh_data = get_client_data(user_id)
    
    # Store in cache with timestamp
    _cache[cache_key] = (current_time, fresh_data)
    
    return fresh_data

# Usage - replace your existing code:
# OLD: client_data = get_client_data(user_id)
# NEW:
user_id = "83906"
client_data = get_cached_client_data(user_id)

# Optional: Clear cache for specific user
def clear_user_cache(user_id: str):
    cache_key = f"client_{user_id}"
    if cache_key in _cache:
        del _cache[cache_key]
        print(f"🗑️ Cleared cache for user {user_id}")

# Optional: Clear all cache
def clear_all_cache():
    _cache.clear()
    print("🗑️ Cleared all cache")