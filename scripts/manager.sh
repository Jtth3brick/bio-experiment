#!/bin/bash
#SBATCH --job-name=train_manager
#SBATCH --account=ac_wolflab
#SBATCH --partition=savio3
#SBATCH --time=72:00:00
#SBATCH --output=logs/manager_%j.out

# This script should be run from a work directory that contains:
# - config.yaml: Configuration file with splits and pipe configs
# - redis.conf: Redis configuration
# - data_getter.py: Implementation of DataGetter interface

# Create directories if they don't exist
mkdir -p logs
mkdir -p data
mkdir -p split_data

# Activate virtual environment with the installed package
source /global/scratch/users/jeshuagustafson/distributed_hyperparam_search/.venv/bin/activate

# Start Redis using Apptainer in the work directory
apptainer exec \
    -B $(pwd)/redis.conf:/usr/local/etc/redis/redis.conf \
    -B $(pwd)/data:/data \
    docker://redis:alpine redis-server /usr/local/etc/redis/redis.conf &

echo "$(hostname):6379" > redis_connection.txt

# Wait for Redis to start
sleep 10

# Run the manager using the installed package
# Assumes data_getter.py exists in the work directory with a DataGetter implementation
python -u -c "
from distributed_hyperparam_search import HyperparamSearchManager
from data_getter import DataGetter as DataGetterImpl

manager = HyperparamSearchManager(
    config_path='config.yaml',
    data_getter=DataGetterImpl(),
    split_dir='split_data',
    redis_host='localhost',
    redis_port=6379
)

# Clear existing queue
manager.clear_queue()

# Setup splits and populate queue
print('Setting up splits and populating Redis queue...')
manager.setup_splits_and_queue()

print(f'Queue size: {manager.get_queue_size()}')
print(f'Results count: {manager.get_results_count()}')
print('Setup complete. Keeping Redis alive...')
"

# Keep Redis alive
sleep 71h

# Make one final save with a different name
apptainer exec docker://redis:alpine redis-cli -h localhost CONFIG SET dbfilename final_save.rdb
apptainer exec docker://redis:alpine redis-cli -h localhost SAVE