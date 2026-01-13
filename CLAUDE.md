# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`abcd-tools` is a Python package for working with ABCD Study data, providing tools for processing fMRI task behavior (ePrime files) and preprocessing neuroimaging data.

## Development Setup

This project uses [Pixi](https://pixi.sh) for dependency management.

```bash
# Clone repository
git clone git@github.com:ajbarrows/abcd-tools
cd abcd-tools

# Install with development environment (recommended)
pixi install -e dev

# Activate shell
pixi shell -e dev

# Or use specific environments:
# - default: Core dependencies only
# - dev: Development tools (testing, linting, docs)
# - modeling: Adds PyMC and PyTorch for predictive modeling
# - matlab: Adds MATLAB integration tools
# - full: All features enabled
```

## Common Commands

### Testing
```bash
# Run all tests with coverage
pixi run test-cov

# Run tests without coverage
pixi run test

# Run specific test file
pytest tests/test_behavior.py

# Run with verbose output
pixi run pytest -v
```

### Code Quality
```bash
# Format code with black
pixi run format

# Lint code with ruff
pixi run lint

# Run pre-commit hooks manually
pixi run pre-commit

# Install pre-commit hooks
pixi run pre-commit install
```

### Building/Packaging
```bash
# Build package
pixi run python -m build
```

## Architecture

### Base Classes (`abcd_tools/base.py`)

The package defines an abstract base class:
- `AbstractDataset`: Interface for loading and processing datasets

Concrete dataset implementations should inherit from this base class.

### Module Organization

**`abcd_tools/task/`**
- `behavior.py`: ePrime file processing for fMRI tasks
  - `EPrimeDataset`: Loads ePrime files with task-specific column selection
  - `EPrimeProcessor`: Processes MID, SST, and nBack task events (alignment, timing conversion)
- `metrics.py`: Computes task performance metrics
  - `DPrimeDataset`: Computes d-prime for nBack task

**`abcd_tools/image/`**
- `preprocess.py`: fMRI preprocessing utilities
  - Vertex-to-ROI mapping with nonzero averaging
  - Beta averaging across runs (weighted by degrees of freedom)
  - Outlier removal and normalization
  - `LinearResidualizer`: Residualize imaging data against covariates
- `transform.py`: Model interpretation tools
  - `haufe_transform()`: Memory-efficient Haufe transformation for interpreting linear model weights

**`abcd_tools/utils/`**
- `config_loader.py`: YAML configuration file loading/saving
- `io.py`: Data loading and manipulation utilities
  - `load_tabular()`: Load CSV with optional column/timepoint filtering
  - `load_dof()`: Load degrees of freedom from ABCD imaging QC files
  - `apply_nda_names()`: Map variable names to NDA conventions
  - `parse_vol_info()`: Parse volume information from ABCD naming conventions
  - `parse_variable_mapping()`: Map variables across ABCD data releases
  - `pd_query_parquet()`: Query and rename columns from parquet files

**`abcd_tools/modeling/`** (New Module - Predictive Modeling)
- `dataset.py`: Loading and managing task fMRI beta estimates
  - `load_betas()`: Load beta estimates from HDF5/MATLAB format with proper indexing
  - `load_task()`: Load full task data structure (conditions × runs × hemispheres)
  - `save_task()`, `load_saved_task()`: Serialize/deserialize task data for faster loading
  - `load_phenotypes()`: Load phenotype data from parquet files with variable renaming
  - `map_id()`: Map subject IDs to match ABCD conventions
  - `make_contrast()`: Create contrasts between task conditions

- `preprocessing.py`: Data preparation pipeline for predictive modeling
  - `combine_hemispheres()`: Merge left/right hemisphere data
  - `combine_runs()`: Average or concatenate data across runs
  - `combine_runs_weighted()`: DOF-weighted run averaging (accounts for motion censoring)
  - `residualize_features()`: Remove covariate effects using OLS regression
  - `prepare_data()`: Full preprocessing pipeline (parcellation → outlier removal → normalization → run combination → residualization)
  - `prepare_all_experiments()`: Batch prepare all experiment configurations
  - `create_analysis_dataset()`: Create ready-to-use (X, y) arrays for modeling
  - `filter_qc()`, `filter_timepoint()`: QC and timepoint filtering utilities

- `models.py`: Machine learning model training and results collection
  - `enet_cv()`: ElasticNet with nested cross-validation using glmnetpy
  - `run_single_experiment()`: Execute single experiment configuration
  - `ExperimentResults`: Container for collecting and storing experiment results
    - Supports both batch and incremental saving
    - Saves summary statistics as parquet/CSV
    - Optionally stores trained model objects

### Configuration System

Task-specific configurations are stored in `conf/` directory as YAML files:
- `task.yaml`: Defines ePrime columns to load for each task (MID, nBack, SST)
- `dprime.yaml`: Specifies behavioral metrics for d-prime computation
- `deap_dict.yaml`, `nda_dict.yaml`: Data dictionaries for ABCD instruments
- `mappings.yaml`: Custom variable mappings

The `config_loader` utility loads these configurations at runtime, enabling flexible column selection and task-specific processing without hardcoding.

### Task Processing Pattern

Processing ePrime behavioral files follows this pattern:

1. **Load**: `EPrimeDataset.load()` reads tab-delimited ePrime files
2. **Configure**: Uses `task.yaml` to select task-specific columns
3. **Process**: `EPrimeProcessor.process()` applies task-specific transformations:
   - Drops pre-dummy-scan trials
   - Aligns timing to scan start (subtract prep time, convert to seconds)
   - Merges cue/stim timings
   - Computes derived variables (RT, duration, accuracy)
4. **Output**: Returns long-format DataFrame with standardized columns

Each task (MID, SST, nBack) has custom processing methods that handle task-specific timing structures.

## Code Style

The project uses:
- **Black** for code formatting (line length: 88)
- **Ruff** for linting (rules: E, F, I)
- **pydoclint** for docstring validation
- **isort** for import sorting (black profile)

Pre-commit hooks enforce these automatically. The CI runs code style checks on all PRs.

### Modeling Workflow

The modeling module enables systematic evaluation of preprocessing strategies for brain-behavior prediction:

1. **Load Data**: Use `load_task()` to load task fMRI beta estimates from HDF5/MATLAB files
2. **Prepare Experiments**: Use `prepare_all_experiments()` to preprocess data across different configurations
   - Experiment configurations specify: normalize (none/before/after), outliers (none/before/after), parcellation (none/destrieux), parcellation_timing (before/after)
3. **Train Models**: Use `enet_cv()` for ElasticNet with nested cross-validation
4. **Collect Results**: Use `ExperimentResults` to track and save results incrementally

Example workflow:
```python
from abcd_tools.modeling import load_task, prepare_all_experiments, enet_cv, ExperimentResults

# Load task data
task_data = load_task('path/to/data/', task='nback', conditions=['0b', '2b'])

# Define experiment grid
experiments = [
    ('none', 'none', 'none'),
    ('before', 'none', 'none'),
    ('after', 'after', 'destrieux', 'after'),
]

# Prepare all configurations
prepare_all_experiments(
    task_betas=task_data,
    qc=qc_data,
    motion=motion_data,
    experiment_grid=experiments,
    save_path='./data/prepared',
    covariates=covariates
)

# Train models and collect results
results = ExperimentResults(save_path='./data/results')
for exp in experiments:
    data = load_prepared_data('./data/prepared', 'nback', '0b', exp)
    X, y = prepare_for_preprocessing(data, phenotypes, outcome='age')
    models, scores = enet_cv(X, y)
    results.save_incremental('nback', '0b', exp, 'age', scores, X.shape[1], X.shape[0])
```

## Testing

- Tests use pytest with coverage reporting
- Coverage configured to omit `tests/*` directory
- CI uploads coverage to Codecov
- Test files follow `test_*.py` naming convention
- Current test coverage: **50%** across core modules
- **Test Suite**: 82 total tests (78 passing, 4 skipped) covering:
  - `config_loader`: YAML I/O operations
  - `behavior`: ePrime file parsing and task-specific processing (MID, nBack, SST)
  - `metrics`: D-prime calculation components (3 tests skipped - require actual ABCD data)
  - `preprocess`: Image preprocessing, outlier removal, normalization, beta averaging, residualization
  - **`modeling.dataset`**: Beta loading, task data structures, phenotype loading, ID mapping
  - **`modeling.preprocessing`**: Hemisphere/run combining, weighted averaging, residualization, QC filtering
  - **`modeling.models`**: ElasticNet CV training with glmnet, experiment results collection

## Current Limitations

- **Documentation**: MkDocs structure exists but content needs completion
- **Test Data**: Some tests (d-prime computation) require actual ABCD data structure and are skipped
