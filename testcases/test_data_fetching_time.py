import time
import logging
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
from datetime import datetime

# Import your database functions
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

def measure_individual_queries(user_id: str) -> Dict[str, float]:
    """
    Measure the time taken by each individual database query.
    """
    print(f"🔍 Measuring individual query times for user: {user_id}")
    print("-" * 60)
    
    times = {}
    
    # Define all queries to test
    queries = {
        'profile': lambda: get_client_profile.invoke(user_id),
        'account_overview': lambda: get_client_account_overview.invoke(user_id),
        'account_aging': lambda: get_client_account_aging.invoke(user_id),
        'banking_details': lambda: get_client_banking_details.invoke(user_id),
        'subscription': lambda: get_client_subscription_amount.invoke(user_id),
        'payment_history': lambda: get_client_payment_history.invoke(user_id),
        'account_statement': lambda: get_client_account_statement.invoke(user_id),
        'debit_mandates': lambda: get_client_debit_mandates.invoke(user_id),
    }
    
    for query_name, query_func in queries.items():
        try:
            start = time.time()
            result = query_func()
            elapsed = time.time() - start
            times[query_name] = elapsed
            
            # Get result info
            result_info = ""
            if isinstance(result, list):
                result_info = f"(returned {len(result)} items)"
            elif isinstance(result, dict):
                result_info = f"(returned dict with {len(result)} keys)"
            elif result is None:
                result_info = "(returned None)"
            else:
                result_info = f"(returned {type(result).__name__})"
            
            print(f"✓ {query_name:20} {elapsed:6.3f}s {result_info}")
            
        except Exception as e:
            times[query_name] = float('inf')  # Mark as failed
            print(f"✗ {query_name:20} FAILED: {str(e)[:50]}...")
    
    print("-" * 60)
    total_sequential = sum(t for t in times.values() if t != float('inf'))
    print(f"📊 Total sequential time: {total_sequential:.3f}s")
    
    return times

def measure_concurrent_performance(user_id: str, max_workers: int = 6) -> Dict[str, Any]:
    """
    Measure concurrent execution performance with different thread counts.
    """
    print(f"\n🚀 Measuring concurrent performance for user: {user_id}")
    print("-" * 60)
    
    queries = {
        'profile': lambda: get_client_profile.invoke(user_id),
        'account_overview': lambda: get_client_account_overview.invoke(user_id),
        'account_aging': lambda: get_client_account_aging.invoke(user_id),
        'banking_details': lambda: get_client_banking_details.invoke(user_id),
        'subscription': lambda: get_client_subscription_amount.invoke(user_id),
        'payment_history': lambda: get_client_payment_history.invoke(user_id),
    }
    
    results = {}
    
    # Test different thread counts
    for workers in [1, 2, 4, 6, 8]:
        print(f"\n🔧 Testing with {workers} workers...")
        
        start_time = time.time()
        failed_tasks = []
        query_times = {}
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks and track individual completion times
            future_to_name = {}
            submit_times = {}
            
            for name, query_func in queries.items():
                submit_time = time.time()
                future = executor.submit(query_func)
                future_to_name[future] = name
                submit_times[name] = submit_time
            
            # Collect results
            for future in as_completed(future_to_name):
                completion_time = time.time()
                task_name = future_to_name[future]
                
                try:
                    result = future.result()
                    task_duration = completion_time - submit_times[task_name]
                    query_times[task_name] = task_duration
                    print(f"  ✓ {task_name} completed in {task_duration:.3f}s")
                except Exception as e:
                    failed_tasks.append(task_name)
                    print(f"  ✗ {task_name} failed: {str(e)[:30]}...")
        
        total_time = time.time() - start_time
        results[workers] = {
            'total_time': total_time,
            'query_times': query_times,
            'failed_tasks': failed_tasks,
            'success_count': len(queries) - len(failed_tasks)
        }
        
        print(f"  📊 Total time: {total_time:.3f}s, Success: {len(queries) - len(failed_tasks)}/{len(queries)}")
    
    return results

def measure_repeated_calls(user_id: str, iterations: int = 5) -> Dict[str, List[float]]:
    """
    Measure performance consistency across multiple calls.
    """
    print(f"\n🔄 Measuring consistency across {iterations} iterations for user: {user_id}")
    print("-" * 60)
    
    queries = {
        'profile': lambda: get_client_profile.invoke(user_id),
        'account_overview': lambda: get_client_account_overview.invoke(user_id),
        'account_aging': lambda: get_client_account_aging.invoke(user_id),
    }
    
    results = {name: [] for name in queries.keys()}
    
    for i in range(iterations):
        print(f"\nIteration {i + 1}/{iterations}")
        for query_name, query_func in queries.items():
            try:
                start = time.time()
                result = query_func()
                elapsed = time.time() - start
                results[query_name].append(elapsed)
                print(f"  {query_name}: {elapsed:.3f}s")
            except Exception as e:
                results[query_name].append(float('inf'))
                print(f"  {query_name}: FAILED")
    
    # Calculate statistics
    print(f"\n📈 Performance Statistics:")
    print("-" * 40)
    for query_name, times in results.items():
        valid_times = [t for t in times if t != float('inf')]
        if valid_times:
            avg_time = statistics.mean(valid_times)
            min_time = min(valid_times)
            max_time = max(valid_times)
            std_dev = statistics.stdev(valid_times) if len(valid_times) > 1 else 0
            
            print(f"{query_name:20}")
            print(f"  Average: {avg_time:.3f}s")
            print(f"  Range:   {min_time:.3f}s - {max_time:.3f}s")
            print(f"  Std Dev: {std_dev:.3f}s")
            print(f"  Success: {len(valid_times)}/{iterations}")
        else:
            print(f"{query_name:20} - All calls failed")
    
    return results

def comprehensive_bottleneck_analysis(user_id: str) -> Dict[str, Any]:
    """
    Run a comprehensive bottleneck analysis.
    """
    print("=" * 80)
    print(f"🔬 COMPREHENSIVE BOTTLENECK ANALYSIS")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👤 User ID: {user_id}")
    print("=" * 80)
    
    analysis_results = {}
    
    # 1. Individual query measurements
    try:
        individual_times = measure_individual_queries(user_id)
        analysis_results['individual_times'] = individual_times
    except Exception as e:
        print(f"❌ Individual query measurement failed: {e}")
        analysis_results['individual_times'] = {}
    
    # 2. Concurrent performance
    try:
        concurrent_results = measure_concurrent_performance(user_id)
        analysis_results['concurrent_results'] = concurrent_results
    except Exception as e:
        print(f"❌ Concurrent measurement failed: {e}")
        analysis_results['concurrent_results'] = {}
    
    # 3. Consistency measurement
    try:
        consistency_results = measure_repeated_calls(user_id, iterations=3)
        analysis_results['consistency_results'] = consistency_results
    except Exception as e:
        print(f"❌ Consistency measurement failed: {e}")
        analysis_results['consistency_results'] = {}
    
    # 4. Generate summary and recommendations
    generate_performance_summary(analysis_results)
    
    return analysis_results

def generate_performance_summary(results: Dict[str, Any]) -> None:
    """
    Generate a summary with actionable recommendations.
    """
    print("\n" + "=" * 80)
    print("📋 PERFORMANCE SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    individual_times = results.get('individual_times', {})
    concurrent_results = results.get('concurrent_results', {})
    
    if individual_times:
        # Find slowest queries
        valid_times = {k: v for k, v in individual_times.items() if v != float('inf')}
        if valid_times:
            slowest = max(valid_times.items(), key=lambda x: x[1])
            fastest = min(valid_times.items(), key=lambda x: x[1])
            total_time = sum(valid_times.values())
            
            print(f"🐌 Slowest query: {slowest[0]} ({slowest[1]:.3f}s)")
            print(f"⚡ Fastest query: {fastest[0]} ({fastest[1]:.3f}s)")
            print(f"⏱️  Total sequential time: {total_time:.3f}s")
            
            # Find bottlenecks (queries taking >50% of total time)
            bottlenecks = [(k, v) for k, v in valid_times.items() if v > total_time * 0.3]
            if bottlenecks:
                print(f"\n🚨 BOTTLENECKS (>30% of total time):")
                for query, time_taken in bottlenecks:
                    percentage = (time_taken / total_time) * 100
                    print(f"   • {query}: {time_taken:.3f}s ({percentage:.1f}%)")
    
    if concurrent_results:
        # Find optimal thread count
        best_performance = min(concurrent_results.items(), key=lambda x: x[1]['total_time'])
        print(f"\n🎯 Optimal thread count: {best_performance[0]} workers")
        print(f"   Best time: {best_performance[1]['total_time']:.3f}s")
        
        # Calculate speedup
        sequential_time = concurrent_results.get(1, {}).get('total_time')
        if sequential_time:
            speedup = sequential_time / best_performance[1]['total_time']
            print(f"   Speedup: {speedup:.1f}x faster than sequential")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("1. Focus optimization on the slowest queries first")
    print("2. Consider database indexing on user_id columns")
    print("3. Implement connection pooling if not already done")
    print("4. Consider caching for frequently accessed data")
    if concurrent_results:
        best_workers = min(concurrent_results.items(), key=lambda x: x[1]['total_time'])[0]
        print(f"5. Use {best_workers} concurrent workers for optimal performance")
    
    print("\n" + "=" * 80)

# Usage example:
def run_analysis(user_id: str = "1489698"):
    """
    Run the complete bottleneck analysis.
    """
    return comprehensive_bottleneck_analysis(user_id)

# Quick individual test function
def quick_test(user_id: str = "1489698"):
    """
    Quick test of just the main queries.
    """
    return measure_individual_queries(user_id)