#!/usr/bin/env python
"""Extract feature importances from top models."""

import os
import pickle
import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_feature_importances(fitted_pipe) -> List[float]:
    """
    Extract feature importances from a fitted pipeline.
    """
    # Get the final Model step
    model = fitted_pipe.named_steps.get("Model") or fitted_pipe.named_steps.get("model")

    if model is None:
        raise ValueError("No Model step found in pipeline")

    # Extract importances
    if hasattr(model, "feature_importances_"):
        # Tree-based models
        importances = list(model.feature_importances_)
    elif hasattr(model, "coef_"):
        # Linear models
        coef = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
        importances = list(np.abs(coef))
    else:
        raise RuntimeError(f"Could not find importances for {fitted_pipe}")

    return importances


def main():
    models_dir = Path("top_models/")
    db_path = "bio.sqlite3"

    # Load feature inputs (CV models only)
    splits = {}

    split_data_dir = Path("../experiment_train/split_data/")
    for fn in os.listdir(split_data_dir):
        path = split_data_dir / fn
        print(f"loading split_data from {path}")
        with open(path, "rb") as f:
            split_data = pickle.load(f)
        splits[str(split_data.split_id)] = np.asarray(split_data.X_cv.columns)

    # Get all top models from database
    conn = sqlite3.connect(db_path)
    top_models = list(
        conn.execute(
            "SELECT DISTINCT config_hash, pipeline_name, split_id FROM top_models"
        )
    )
    conn.close()

    print(f"Found {len(top_models)} models in top_models view")

    all_results = []
    skipped = []

    # Extract features from each model
    for config_hash, pipeline_name, split_id in tqdm(
        top_models, desc="Extracting features"
    ):
        model_path = models_dir / f"{config_hash}.model.pkl"

        if not model_path.exists():
            print(f"Warning: Model file not found for {config_hash}")
            continue

        # Load model
        with open(model_path, "rb") as f:
            fitted_pipe = pickle.load(f)

        if any(
            "MLP" in type(step).__name__ or "SVC" in type(step).__name__
            for step in fitted_pipe.named_steps.values()
        ):
            skipped.append(fitted_pipe)
            continue

        feature_names = list(
            fitted_pipe[:-1].get_feature_names_out(splits[str(split_id)])
        )

        if not feature_names:
            raise RuntimeError(f"{feature_names=} is not valid")

        importances = extract_feature_importances(fitted_pipe)
        if not importances:
            raise RuntimeError(f"{importances=} is not valid")

        if len(importances) != len(feature_names):
            raise RuntimeError(
                f"{len(importances)=} is not equal to {len(feature_names)=} for {()}"
            )

        # Store results
        for fname, imp in zip(feature_names, importances):
            all_results.append(
                {
                    "config_hash": config_hash,
                    "pipeline_name": pipeline_name,
                    "feature": fname,
                    "importance": float(imp),
                }
            )

    # Save to database
    if all_results:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()

        # Create unique index on config_hash in train_results if it doesn't exist
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_train_results_config_hash
            ON train_results(config_hash)
        """)

        # Drop table if exists
        cursor.execute("DROP TABLE IF EXISTS feature_importances")

        # Create table with foreign keys
        cursor.execute("""
            CREATE TABLE feature_importances (
                config_hash TEXT NOT NULL,
                taxon_id INTEGER NOT NULL,
                importance REAL NOT NULL,
                FOREIGN KEY (config_hash) REFERENCES train_results(config_hash),
                FOREIGN KEY (taxon_id) REFERENCES taxa(taxon_id),
                PRIMARY KEY (config_hash, taxon_id)
            )
        """)

        # Insert data - cast feature to int for taxon_id
        cursor.executemany(
            "INSERT INTO feature_importances (config_hash, taxon_id, importance) VALUES (?, ?, ?)",
            [(r["config_hash"], int(r["feature"]), r["importance"]) for r in all_results]
        )

        conn.commit()
        conn.close()

        print(f"\nSaved {len(all_results)} feature importances to database")
        print(f"Extracted from {len(set(r['config_hash'] for r in all_results))} models")
    else:
        print("\nNo feature importances extracted")

    print(f"Skipped {len(skipped)} models.")


if __name__ == "__main__":
    main()
