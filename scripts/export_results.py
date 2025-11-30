#!/usr/bin/env python
"""
Export all completed ModelConfig objects from Redis to a pickle file.
This preserves the full objects including fitted pipelines if stored.
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import List

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


def export_results(
    r: redis.Redis,
    output_file: str = None,
    include_failed: bool = False,
    verbose: bool = True
) -> List:
    """
    Export all completed ModelConfig objects from Redis.
    
    Args:
        r: Redis connection
        output_file: Output pickle file path (default: results_YYYYMMDD_HHMMSS.pkl)
        include_failed: Whether to include models that timed out or failed
        verbose: Print progress information
        
    Returns:
        List of ModelConfig objects
    """
    
    # Generate default filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results_{timestamp}.pkl"
    
    # Get all result hashes
    result_hashes = r.smembers("results")
    
    if not result_hashes:
        print("No results found in Redis.")
        return []
    
    if verbose:
        print(f"Found {len(result_hashes)} results in Redis")
        print(f"Exporting to: {output_file}")
    
    # Collect all ModelConfig objects
    model_configs = []
    successful = 0
    failed = 0
    
    # Use tqdm for progress bar
    iterator = tqdm(result_hashes, desc="Exporting results") if verbose else result_hashes
    
    for result_hash in iterator:
        # Decode hash if it's bytes
        if isinstance(result_hash, bytes):
            hash_str = result_hash.decode('utf-8')
        else:
            hash_str = result_hash
        
        result_data = r.get(f"result:{hash_str}")
        if result_data:
            try:
                model_config = pickle.loads(result_data)
                
                # Check if it's a successful model
                has_scores = bool(model_config.cv_scores or model_config.validate_score is not None)
                
                if has_scores:
                    successful += 1
                    model_configs.append(model_config)
                elif include_failed:
                    failed += 1
                    model_configs.append(model_config)
                else:
                    failed += 1
                    
            except Exception as e:
                if verbose:
                    print(f"\nWarning: Failed to load result {hash_str}: {e}")
    
    # Save to pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(model_configs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Print summary
    if verbose:
        print(f"\n" + "="*70)
        print("EXPORT COMPLETE")
        print("="*70)
        print(f"Total results exported: {len(model_configs)}")
        print(f"  Successful models: {successful}")
        if include_failed:
            print(f"  Failed/timeout models: {failed}")
        else:
            print(f"  Failed/timeout models (skipped): {failed}")
        print(f"Output file: {output_file}")
        print(f"File size: {Path(output_file).stat().st_size / (1024*1024):.2f} MB")
        
        # Show sample of what was exported
        if model_configs:
            print(f"\nSample of exported models:")
            print("-" * 40)
            for i, model in enumerate(model_configs[:3]):
                print(f"  {i+1}. Pipeline: {model.pipeline_name}, Split: {model.split_id}")
                if model.cv_scores:
                    import numpy as np
                    print(f"     CV Score: {np.mean(model.cv_scores):.4f} ± {np.std(model.cv_scores):.4f}")
                if model.validate_score is not None:
                    print(f"     Validation Score: {model.validate_score:.4f}")
    
    return model_configs


def load_results(pickle_file: str, verbose: bool = True):
    """
    Load ModelConfig objects from a pickle file.
    
    Args:
        pickle_file: Path to pickle file
        verbose: Print information about loaded data
        
    Returns:
        List of ModelConfig objects
    """
    with open(pickle_file, 'rb') as f:
        model_configs = pickle.load(f)
    
    if verbose:
        print(f"Loaded {len(model_configs)} models from {pickle_file}")
        
        # Analyze what was loaded
        by_pipeline = {}
        by_split = {}
        scores = []
        
        for model in model_configs:
            # Count by pipeline
            by_pipeline[model.pipeline_name] = by_pipeline.get(model.pipeline_name, 0) + 1
            
            # Count by split
            by_split[model.split_id] = by_split.get(model.split_id, 0) + 1
            
            # Collect scores
            if model.cv_scores:
                import numpy as np
                scores.append(np.mean(model.cv_scores))
        
        print(f"\nModels by pipeline:")
        for pipeline, count in sorted(by_pipeline.items()):
            print(f"  {pipeline}: {count}")
        
        print(f"\nModels by split:")
        for split, count in sorted(by_split.items()):
            print(f"  Split {split}: {count}")
        
        if scores:
            import numpy as np
            scores_array = np.array(scores)
            print(f"\nScore statistics:")
            print(f"  Mean: {scores_array.mean():.4f}")
            print(f"  Std: {scores_array.std():.4f}")
            print(f"  Min: {scores_array.min():.4f}")
            print(f"  Max: {scores_array.max():.4f}")
    
    return model_configs


def main():
    parser = argparse.ArgumentParser(
        description="Export completed ModelConfig objects from Redis to pickle file"
    )
    parser.add_argument(
        "--redis-connection",
        type=str,
        default="redis_connection.txt",
        help="Redis connection file (default: redis_connection.txt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output pickle file (default: results_YYYYMMDD_HHMMSS.pkl)"
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include models that timed out or failed"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Load and analyze a previously exported pickle file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()
    
    # If loading an existing file
    if args.load:
        load_results(args.load, verbose=not args.quiet)
    else:
        # Connect to Redis and export
        r = connect_redis(args.redis_connection)
        export_results(
            r,
            output_file=args.output,
            include_failed=args.include_failed,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    # Add package to path if needed
    package_path = Path("/global/scratch/users/jeshuagustafson/distributed_hyperparam_search")
    if package_path.exists() and str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))
    
    main()