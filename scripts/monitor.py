#!/usr/bin/env python
"""
Live monitor for distributed hyperparameter search with tqdm progress bar.
Shows real-time queue progress and model training statistics.
"""

import argparse
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import redis
from tqdm import tqdm


def connect_redis(connection_file: str = "redis_connection.txt") -> redis.Redis:
    """Connect to Redis using connection file"""
    try:
        with open(connection_file, "r") as f:
            host, port = f.read().strip().split(":")
        r = redis.Redis(host=host, port=int(port), decode_responses=False)
        r.ping()  # Test connection
        return r
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        print("Make sure the manager job is running and Redis is active.")
        sys.exit(1)


def monitor_queue(r: redis.Redis, refresh_interval: int = 5):
    """Monitor queue progress with tqdm progress bar"""
    
    # Get initial counts
    initial_queue = r.llen("model_queue")
    initial_results = r.scard("results")
    total_tasks = initial_queue + initial_results
    
    if total_tasks == 0:
        print("No tasks found in queue or results.")
        return
    
    # Track start time and completed tasks
    start_time = datetime.now()
    initial_completed = initial_results
    
    # Create progress bar
    pbar = tqdm(
        total=total_tasks,
        initial=initial_results,
        desc="Training Models",
        unit="models",
        bar_format="{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ncols=100
    )
    
    # Additional stats tracking
    last_update = initial_results
    stats = {
        'successful': 0,
        'failed': 0,
        'by_pipeline': {},
        'by_split': {},
        'recent_scores': []
    }
    
    try:
        while True:
            # Get current counts
            queue_remaining = r.llen("model_queue")
            results_count = r.scard("results")
            
            # Update progress bar
            new_completed = results_count - last_update
            if new_completed > 0:
                pbar.update(new_completed)
                last_update = results_count
            
            # Calculate statistics
            completed_this_session = results_count - initial_completed
            if completed_this_session > 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = completed_this_session / elapsed if elapsed > 0 else 0
                eta = timedelta(seconds=queue_remaining / rate) if rate > 0 else None
            else:
                rate = 0
                eta = None
            
            # Sample recent results for score statistics
            if results_count > stats['successful'] + stats['failed']:
                result_hashes = r.smembers("results")
                sample_size = min(100, len(result_hashes))  # Sample last 100
                
                stats['successful'] = 0
                stats['failed'] = 0
                stats['by_pipeline'].clear()
                stats['by_split'].clear()
                stats['recent_scores'].clear()
                
                for i, result_hash in enumerate(result_hashes):
                    if i >= sample_size:
                        break
                    
                    # Decode hash if bytes
                    if isinstance(result_hash, bytes):
                        hash_str = result_hash.decode('utf-8')
                    else:
                        hash_str = result_hash
                    
                    result_data = r.get(f"result:{hash_str}")
                    if result_data:
                        try:
                            model_config = pickle.loads(result_data)
                            
                            # Count success/failure
                            if model_config.cv_scores or model_config.validate_score is not None:
                                stats['successful'] += 1
                                
                                # Track by pipeline
                                pipeline = model_config.pipeline_name
                                stats['by_pipeline'][pipeline] = stats['by_pipeline'].get(pipeline, 0) + 1
                                
                                # Track by split
                                split = model_config.split_id
                                stats['by_split'][split] = stats['by_split'].get(split, 0) + 1
                                
                                # Track scores
                                if model_config.cv_scores:
                                    import numpy as np
                                    stats['recent_scores'].append(np.mean(model_config.cv_scores))
                            else:
                                stats['failed'] += 1
                        except:
                            pass
            
            # Clear screen and print stats
            print("\033[2J\033[H")  # Clear screen and move cursor to top
            print("=" * 100)
            print("DISTRIBUTED HYPERPARAMETER SEARCH MONITOR")
            print("=" * 100)
            print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Current: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if eta:
                print(f"ETA: {(datetime.now() + eta).strftime('%Y-%m-%d %H:%M:%S')} (approx)")
            print()
            
            # Progress bar
            pbar.display()
            print()
            
            # Queue stats
            print(f"Queue Statistics:")
            print(f"  Remaining in queue: {queue_remaining:,}")
            print(f"  Completed total: {results_count:,}")
            print(f"  Completed this session: {completed_this_session:,}")
            print(f"  Rate: {rate:.2f} models/second")
            print()
            
            # Success/failure from sample
            if stats['successful'] + stats['failed'] > 0:
                success_rate = stats['successful'] / (stats['successful'] + stats['failed']) * 100
                print(f"Sample Statistics (last {sample_size}):")
                print(f"  Successful: {stats['successful']} ({success_rate:.1f}%)")
                print(f"  Failed/Timeout: {stats['failed']} ({100-success_rate:.1f}%)")
                
                if stats['recent_scores']:
                    import numpy as np
                    scores_array = np.array(stats['recent_scores'])
                    print(f"  Recent CV scores: {np.mean(scores_array):.4f} ± {np.std(scores_array):.4f}")
                
                # Pipeline distribution
                if stats['by_pipeline']:
                    print("\n  Models by pipeline:")
                    for pipeline, count in sorted(stats['by_pipeline'].items(), key=lambda x: -x[1])[:5]:
                        print(f"    {pipeline}: {count}")
                
                # Split distribution
                if stats['by_split']:
                    print("\n  Models by split:")
                    for split, count in sorted(stats['by_split'].items()):
                        print(f"    Split {split}: {count}")
            
            # Check if complete
            if queue_remaining == 0:
                print("\n✅ All tasks completed!")
                pbar.close()
                break
            
            # Wait before next update
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        pbar.close()
        print("\n\nMonitoring stopped by user.")
    except Exception as e:
        pbar.close()
        print(f"\n\nError during monitoring: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Live monitor for distributed hyperparameter search"
    )
    parser.add_argument(
        "--redis-connection",
        type=str,
        default="redis_connection.txt",
        help="Redis connection file (default: redis_connection.txt)"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=5,
        help="Refresh interval in seconds (default: 5)"
    )
    args = parser.parse_args()
    
    # Connect to Redis
    r = connect_redis(args.redis_connection)
    
    # Start monitoring
    monitor_queue(r, args.refresh)


if __name__ == "__main__":
    # Add package to path if needed
    package_path = Path("/global/scratch/users/jeshuagustafson/distributed_hyperparam_search")
    if package_path.exists() and str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))
    
    main()