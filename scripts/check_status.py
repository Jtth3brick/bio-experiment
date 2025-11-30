#!/usr/bin/env python
"""
Check the status of a distributed hyperparameter search experiment.
Run from within a work directory to see progress and export results.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import redis


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


def print_split_statistics(split_dir: str = "split_data"):
    """Print statistics about the split data files"""
    split_path = Path(split_dir)
    if not split_path.exists():
        print(f"Split directory not found: {split_dir}")
        return

    print("\n" + "=" * 70)
    print("SPLIT DATA STATISTICS")
    print("=" * 70)

    split_files = sorted(split_path.glob("split_*.pkl"))

    for split_file in split_files:
        with open(split_file, "rb") as f:
            split_config = pickle.load(f)

        split_id = split_config.split_id
        print(f"\nSplit {split_id}:")
        print("-" * 30)

        if split_config.X_train is not None:
            print(
                f"  Train set: {split_config.X_train.shape[0]} samples, {split_config.X_train.shape[1]} features"
            )
            print(f"  Train labels: {split_config.y_train.value_counts().to_dict()}")
        else:
            print("  Train set: Not configured")

        if split_config.X_val is not None:
            print(f"  Validation set: {split_config.X_val.shape[0]} samples")
            print(f"  Validation labels: {split_config.y_val.value_counts().to_dict()}")
        else:
            print("  Validation set: Not configured")

        if split_config.X_cv is not None:
            print(f"  CV set: {split_config.X_cv.shape[0]} samples")
            print(f"  CV folds: {len(split_config.cv_indices)}")
            print(f"  CV labels: {split_config.y_cv.value_counts().to_dict()}")
        else:
            print("  CV set: Not configured")


def print_queue_statistics(r: redis.Redis):
    """Print Redis queue statistics"""
    print("\n" + "=" * 70)
    print("QUEUE STATISTICS")
    print("=" * 70)

    queue_size = r.llen("model_queue")
    results_count = r.scard("results")

    print(f"\nQueue Status:")
    print(f"  Remaining in queue: {queue_size}")
    print(f"  Completed results: {results_count}")

    if results_count > 0:
        total_tasks = queue_size + results_count
        completion_pct = (results_count / total_tasks) * 100
        print(f"  Total tasks: {total_tasks}")
        print(f"  Completion: {completion_pct:.1f}%")

    # Check for any failed models (models with no scores)
    result_hashes = r.smembers("results")
    failed_count = 0

    for result_hash in result_hashes:
        # Decode hash if it's bytes
        if isinstance(result_hash, bytes):
            hash_str = result_hash.decode("utf-8")
        else:
            hash_str = result_hash

        result_data = r.get(f"result:{hash_str}")
        if result_data:
            model_config = pickle.loads(result_data)
            if not model_config.cv_scores and model_config.validate_score is None:
                failed_count += 1

    if failed_count > 0:
        print(f"  Failed/Timeout: {failed_count}")


def export_results_to_csv(r: redis.Redis, output_file: str = "results.csv"):
    """Export all completed results to CSV"""
    print("\n" + "=" * 70)
    print("EXPORTING RESULTS")
    print("=" * 70)

    result_hashes = r.smembers("results")

    if not result_hashes:
        print("No results to export yet.")
        return

    results_data = []

    for result_hash in result_hashes:
        # Decode hash if it's bytes
        if isinstance(result_hash, bytes):
            hash_str = result_hash.decode("utf-8")
        else:
            hash_str = result_hash

        result_data = r.get(f"result:{hash_str}")
        if result_data:
            model_config = pickle.loads(result_data)

            # Extract hyperparameters
            hyperparams = model_config._unfit_pipe.get_params()

            # Create base record
            record = {
                "config_hash": model_config.config_hash,
                "split_id": model_config.split_id,
                "pipeline_name": model_config.pipeline_name,
                "validate_score": model_config.validate_score,
            }

            # Add CV scores
            if model_config.cv_scores:
                record["cv_mean"] = np.mean(model_config.cv_scores)
                record["cv_std"] = np.std(model_config.cv_scores)
                for i, score in enumerate(model_config.cv_scores):
                    record[f"cv_fold_{i}"] = score
            else:
                record["cv_mean"] = None
                record["cv_std"] = None

            # Add key hyperparameters
            for key, value in hyperparams.items():
                if not key.startswith("memory") and not key.startswith("_"):
                    # Simplify nested parameters
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            record[f"hp_{key}_{sub_key}"] = str(sub_value)
                    else:
                        record[f"hp_{key}"] = str(value)

            results_data.append(record)

    # Create DataFrame and save
    df = pd.DataFrame(results_data)

    if len(df) == 0:
        print("\nNo valid results to export (all models may have timed out).")
        return

    # Sort by best score
    if "cv_mean" in df.columns:
        df = df.sort_values("cv_mean", ascending=False, na_position="last")
    elif "validate_score" in df.columns:
        df = df.sort_values("validate_score", ascending=False, na_position="last")

    df.to_csv(output_file, index=False)
    print(f"\nExported {len(df)} results to {output_file}")

    # Print top performers
    print("\nTop 10 Models by Score:")
    print("-" * 70)

    if "cv_mean" in df.columns and df["cv_mean"].notna().any():
        top_models = df.nlargest(10, "cv_mean")[
            ["pipeline_name", "split_id", "cv_mean", "cv_std", "validate_score"]
        ]
        print("\nBy CV Mean Score:")
        print(top_models.to_string(index=False))

    if "validate_score" in df.columns and df["validate_score"].notna().any():
        top_models = df.nlargest(10, "validate_score")[
            ["pipeline_name", "split_id", "validate_score"]
        ]
        print("\nBy Validation Score:")
        print(top_models.to_string(index=False))

    # Print score distributions by pipeline
    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTIONS BY PIPELINE")
    print("=" * 70)

    if "pipeline_name" not in df.columns:
        print("No pipeline information available.")
        return

    for pipeline in df["pipeline_name"].unique():
        pipeline_df = df[df["pipeline_name"] == pipeline]
        print(f"\n{pipeline}:")

        if "cv_mean" in pipeline_df.columns:
            cv_scores = pipeline_df["cv_mean"].dropna()
            if len(cv_scores) > 0:
                print(f"  CV Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                print(f"  CV Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")

        if "validate_score" in pipeline_df.columns:
            val_scores = pipeline_df["validate_score"].dropna()
            if len(val_scores) > 0:
                print(
                    f"  Validation Mean: {val_scores.mean():.4f} ± {val_scores.std():.4f}"
                )
                print(
                    f"  Validation Range: [{val_scores.min():.4f}, {val_scores.max():.4f}]"
                )

        print(f"  Models trained: {len(pipeline_df)}")


def main():
    parser = argparse.ArgumentParser(
        description="Check status of distributed hyperparameter search experiment"
    )
    parser.add_argument(
        "--redis-connection",
        type=str,
        default="redis_connection.txt",
        help="Redis connection file (default: redis_connection.txt)",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default="split_data",
        help="Directory with split data (default: split_data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.csv",
        help="Output CSV file for results (default: results.csv)",
    )
    parser.add_argument("--no-export", action="store_true", help="Skip CSV export")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("DISTRIBUTED HYPERPARAMETER SEARCH STATUS")
    print("=" * 70)
    print(f"Work Directory: {Path.cwd()}")

    # Print split statistics
    print_split_statistics(args.split_dir)

    # Connect to Redis & print queue statistics
    try:
        r = connect_redis(args.redis_connection)
        print_queue_statistics(r)

        # Export results to CSV
        if not args.no_export:
            export_results_to_csv(r, args.output)
    except Exception as e:
        print(f"Failed with redis: {e}")

    print("\n" + "=" * 70)
    print("STATUS CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Add package to path if needed
    package_path = Path(
        "/global/scratch/users/jeshuagustafson/distributed_hyperparam_search"
    )
    if package_path.exists() and str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))

    main()
