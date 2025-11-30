#!/bin/bash
#SBATCH --job-name=train_workers
#SBATCH --account=fc_wolflab
#SBATCH --partition=savio4_htc
#SBATCH --array=0-1499
#SBATCH --time=36:00:00
#SBATCH --output=logs/train_worker_%a_%j.out

# This script should be run from the same work directory as manager.sh
# It expects:
# - redis_connection.txt: Created by manager.sh
# - split_data/: Directory with split pickle files created by manager

# Activate virtual environment with the installed package
source /global/scratch/users/jeshuagustafson/distributed_hyperparam_search/.venv/bin/activate

# Run worker using the installed package
python -u -c "
from distributed_hyperparam_search import Worker

worker = Worker(
    worker_id=$SLURM_ARRAY_TASK_ID,
    split_dir='split_data',
    redis_connection_file='redis_connection.txt',
    skip_completed=True
)

worker.run()
"
