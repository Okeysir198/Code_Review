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

# Execute and return client_data at module level
async def _load_client_data(user_id):
    """Load client data at module level."""
    return await get_client_data_async_cache(user_id)

user_id = "83906"  # Example user ID, can be changed as needed
client_data = asyncio.run(_load_client_data(user_id))
logger.info(f"✅ Client data loaded at module level for user: {client_data.get('user_id', 'unknown')}")

