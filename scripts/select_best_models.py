#!/usr/bin/env python
"""
Select the best model for each pipeline-split combination and save them.
"""

import pickle
import sys
from pathlib import Path
import redis
import numpy as np
from collections import defaultdict

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
        sys.exit(1)


def select_best_models(work_dir: str = ".", output_dir: str = "../experiment_test_selected/data"):
    """
    Select best model for each pipeline-split combination.
    """
    
    # Connect to Redis
    r = connect_redis(f"{work_dir}/redis_connection.txt")
    
    # Get all result hashes
    result_hashes = r.smembers("results")
    
    if not result_hashes:
        print("No results found in Redis.")
        return
    
    print(f"Found {len(result_hashes)} total results in Redis")
    
    # Organize models by pipeline and split
    models_by_key = defaultdict(list)
    
    for result_hash in result_hashes:
        # Decode hash if it's bytes
        if isinstance(result_hash, bytes):
            hash_str = result_hash.decode('utf-8')
        else:
            hash_str = result_hash
        
        result_data = r.get(f"result:{hash_str}")
        if result_data:
            try:
                model_config = pickle.loads(result_data)
                
                # Only consider models with scores
                if model_config.cv_scores:
                    key = (model_config.pipeline_name, model_config.split_id)
                    cv_mean = np.mean(model_config.cv_scores)
                    models_by_key[key].append((cv_mean, model_config))
            except Exception as e:
                print(f"Warning: Failed to load result {hash_str}: {e}")
    
    # Select best model for each key
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    selected_models = {}
    for (pipeline, split), models in models_by_key.items():
        if models:
            # Sort by CV score (descending) and take the best
            models.sort(key=lambda x: x[0], reverse=True)
            best_score, best_model = models[0]
            
            # Save the model
            filename = f"{output_dir}/model_{pipeline}_{split}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(best_model, f)
            
            selected_models[(pipeline, split)] = (best_score, filename)
            print(f"Selected {pipeline} for split {split}: CV={best_score:.4f} ({len(models)} candidates)")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SELECTED MODELS SUMMARY")
    print(f"{'='*60}")
    
    # Group by pipeline
    by_pipeline = defaultdict(list)
    for (pipeline, split), (score, _) in selected_models.items():
        by_pipeline[pipeline].append((split, score))
    
    for pipeline in sorted(by_pipeline.keys()):
        splits_scores = by_pipeline[pipeline]
        splits_scores.sort(key=lambda x: x[0])  # Sort by split ID
        scores = [s for _, s in splits_scores]
        print(f"\n{pipeline}:")
        for split, score in splits_scores:
            print(f"  Split {split}: {score:.4f}")
        print(f"  Average: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    print(f"\nTotal models saved: {len(selected_models)}")
    print(f"Output directory: {output_dir}")
    
    return selected_models


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Select best models per pipeline-split combination")
    parser.add_argument("--work-dir", default=".", help="Work directory with redis connection")
    parser.add_argument("--output-dir", default="../experiment_test_selected/data", help="Output directory")
    args = parser.parse_args()
    
    # Add package to path
    package_path = Path("/global/scratch/users/jeshuagustafson/distributed_hyperparam_search")
    if package_path.exists() and str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))
    
    select_best_models(args.work_dir, args.output_dir)