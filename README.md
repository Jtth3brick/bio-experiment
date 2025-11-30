# Distributed Hyperparameter Search - Training Setup

This directory contains configuration and scripts for running distributed hyperparameter search experiments using the `bio-sklearn-trainer` package (included as a git submodule).

## Directory Structure

```
train/
├── config.yaml                 # Main configuration file with splits and pipelines
├── redis.conf                  # Redis configuration
├── read_db.sqlite3             # SQLite database with training data
├── scripts/                    # SLURM and setup scripts
│   ├── manager.sh             # SLURM script for manager node
│   ├── start_workers.sh       # SLURM script for worker array
│   └── setup_workdir.sh       # Creates new work directories
└── examples/                   # Example Python scripts
    ├── example_run_manager.py  # Python example for running manager
    ├── example_run_worker.py   # Python example for running worker
    └── example_sqlite_data_getter.py  # Example DataGetter implementation
```

## Quick Start

### 1. Clone and Install the Package

```bash
git clone --recurse-submodules <this-repo-url>
cd train-new/bio-sklearn-trainer
uv venv
uv pip install -e ".[ml,dev]"
```

### 2. Create a Work Directory

```bash
cd /global/scratch/users/jeshuagustafson/train
./scripts/setup_workdir.sh my_experiment
```

This creates a new directory with all necessary files:
- `config.yaml` - Configuration (copied from train/)
- `redis.conf` - Redis configuration  
- `manager.sh` & `start_workers.sh` - SLURM scripts
- `data_getter.py` - DataGetter implementation for your data
- `logs/`, `data/`, `split_data/` - Working directories

### 3. Run the Experiment

```bash
cd my_experiment

# Edit config.yaml if needed
vi config.yaml

# Submit manager job (starts Redis and sets up queue)
sbatch manager.sh

# Submit worker array (processes queue)
sbatch start_workers.sh
```

## Configuration

### config.yaml

The configuration file defines:
- **seed**: Random seed for reproducibility
- **num_cv_splits**: Number of cross-validation folds (0 to skip)
- **train_eval**: Whether to do train/validation split
- **model_caching**: Settings for sklearn model caching
- **splits**: Data splits with train/validate cohorts
- **pipe_configs**: Pipeline configurations with hyperparameters

### Pipeline Components

Available transformers (in `distributed_hyperparam_search.custom_transformers`):
- `ThresholdApplier` - Feature selection by threshold
- `RandomForestFeatureSelector` - RF-based feature selection
- `LassoFeatureSelector` - Lasso-based feature selection

Standard sklearn components are also supported.

## SLURM Scripts

### manager.sh
- Runs on `savio3` partition for 72 hours
- Starts Redis server in a container
- Activates the package virtual environment
- Runs the manager to populate task queue
- Keeps Redis alive for workers

### start_workers.sh  
- Runs on `savio4_htc` partition as array job (0-999)
- Each worker processes tasks from Redis queue
- Automatically skips completed tasks
- Runs for 24 hours per worker

## Monitoring

### Check Queue Status
```bash
# Connect to Redis from manager node
apptainer exec docker://redis:alpine redis-cli -h localhost

# Check queue size
LLEN model_queue

# Check completed results
SCARD results
```

### View Logs
```bash
# Manager log
tail -f logs/manager_*.out

# Worker logs
tail -f logs/train_worker_*_*.out
```

## Custom Data Sources

To use a different data source, modify `data_getter.py` in your work directory:

```python
from distributed_hyperparam_search import DataGetter as BaseDataGetter

class DataGetter(BaseDataGetter):
    def get_data(self, cohorts, schema=None):
        # Your data loading logic
        return X, y
```

## Advanced Usage

### Python API

Instead of SLURM scripts, you can use the Python API directly:

```python
from distributed_hyperparam_search import HyperparamSearchManager, Worker
from data_getter import DataGetter

# Run manager
manager = HyperparamSearchManager(
    config_path='config.yaml',
    data_getter=DataGetter(),
    split_dir='split_data'
)
manager.setup_splits_and_queue()

# Run worker
worker = Worker(
    worker_id=0,
    split_dir='split_data'
)
worker.run()
```

### Multiple Experiments

Each work directory is independent, allowing parallel experiments:

```bash
./scripts/setup_workdir.sh experiment_1
./scripts/setup_workdir.sh experiment_2

# Run both experiments
(cd experiment_1 && sbatch manager.sh && sbatch start_workers.sh)
(cd experiment_2 && sbatch manager.sh && sbatch start_workers.sh)
```

## Package Development

To modify the package:

```bash
cd /global/scratch/users/jeshuagustafson/distributed_hyperparam_search
source .venv/bin/activate

# Make changes to the package
vi distributed_hyperparam_search/manager.py

# Changes are immediately available (installed in editable mode)
```

## Troubleshooting

1. **Redis connection error**: Check that manager job is running and Redis started successfully
2. **Import errors**: Ensure the package venv is activated in scripts
3. **Queue empty**: Check manager log to ensure splits were created successfully
4. **Workers not processing**: Check Redis connection file exists in work directory

## Support

For issues with the package, see:
`/global/scratch/users/jeshuagustafson/distributed_hyperparam_search/README.md`
