#!/bin/bash
# Script to set up a new work directory for distributed hyperparameter search

WORKDIR=${1:-"workdir_$(date +%Y%m%d_%H%M%S)"}
PACKAGE_PATH="/global/scratch/users/jeshuagustafson/distributed_hyperparam_search"
TRAIN_PATH="/global/scratch/users/jeshuagustafson/train"
SCRIPTS_PATH="$TRAIN_PATH/scripts"

echo "Setting up work directory: $WORKDIR"

# Create work directory
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Create necessary subdirectories
mkdir -p logs
mkdir -p data
mkdir -p split_data

# Copy required files from train directory
echo "Copying configuration and scripts..."
cp "$TRAIN_PATH/config.yaml" .
cp "$TRAIN_PATH/redis.conf" .
cp "$SCRIPTS_PATH/manager.sh" .
cp "$SCRIPTS_PATH/start_workers.sh" .
cp "$SCRIPTS_PATH/monitor.py" .
cp "$SCRIPTS_PATH/check_status.py" .
cp "$SCRIPTS_PATH/export_results.py" .

# Create data_getter.py from example
echo "Creating data_getter.py..."
cat > data_getter.py << 'EOF'
"""Data getter implementation for this work directory"""

import sqlite3
from typing import List, Optional, Tuple

import pandas as pd

from distributed_hyperparam_search import DataGetter as BaseDataGetter


class DataGetter(BaseDataGetter):
    """SQLite implementation of DataGetter interface"""
    
    def __init__(self, db_path: str = "/global/scratch/users/jeshuagustafson/train/read_db.sqlite3"):
        """Initialize with database path"""
        self.db_path = db_path
    
    def get_data(
        self, 
        cohorts: List[str], 
        schema: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Retrieve the dataset for given cohorts"""
        # Connect to database as read only
        engine = sqlite3.connect(self.db_path)
        engine.execute("PRAGMA query_only = 1")
        
        cohort_list_str = "', '".join(cohorts)
        
        # SQL query to get all nonzero read counts
        query_seq = f""" 
            SELECT g.run, g.taxon_id AS taxon, g.rpm
            FROM genomic_sequence_rpm AS g
            INNER JOIN selected_runs AS s
              ON g.run = s.run
            WHERE s.cohort IN ('{cohort_list_str}')
        """
        
        # Read data into DataFrame
        df = pd.read_sql(query_seq, engine)
        
        # Pivot to get features as columns
        X = df.pivot(index="run", columns="taxon", values="rpm").fillna(0)
        X.columns = [str(col) for col in X.columns]
        
        # Apply schema if provided
        if schema is not None:
            schema = [str(taxon) for taxon in schema]
            
            # Create DataFrame with missing columns
            missing_data = pd.DataFrame(
                0,
                index=X.index,
                columns=list(set(schema) - set(X.columns)),
            )
            
            # Combine and reorder
            X = pd.concat([X, missing_data], axis=1)
            X = X[schema]
        
        # Get labels
        query_labels = f"""
            SELECT run, 
                   CASE WHEN condition = 'Healthy' THEN 0 ELSE 1 END as label
            FROM selected_runs
            WHERE cohort IN ('{cohort_list_str}')
        """
        y = pd.read_sql(query_labels, engine).set_index("run")["label"]
        
        # Ensure same indices
        y = y.reindex(X.index)
        
        # Close connection
        engine.close()
        
        # Ensure correct types
        X_final: pd.DataFrame = X.astype(float)
        y_final: pd.Series = y.astype(int)
        
        return X_final, y_final
EOF

echo ""
echo "Work directory setup complete at: $(pwd)"
echo ""
echo "Directory contents:"
ls -la
echo ""
echo "To use this work directory:"
echo "1. cd $WORKDIR"
echo "2. Edit config.yaml if needed"
echo "3. Submit manager job: sbatch manager.sh"
echo "4. Submit worker jobs: sbatch start_workers.sh"
echo ""
echo "The scripts will use the installed package at:"
echo "  $PACKAGE_PATH"