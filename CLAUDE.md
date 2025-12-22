# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`abcd-tools` is a Python package for working with ABCD Study data, providing tools for processing fMRI task behavior (ePrime files) and preprocessing neuroimaging data.

## Development Setup

```bash
# Clone repository
git clone git@github.com:ajbarrows/abcd-tools
cd abcd-tools

# Option 1: Using conda/mamba (recommended for full environment)
mamba env update -f environment.yml
conda activate abcd-tools
python -m pip install -e .

# Option 2: Using pip only (minimal setup for core functionality)
pip install -e .
```

## Common Commands

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

### Code Quality
```bash
# Run pre-commit hooks manually
pre-commit run --all-files

# Install pre-commit hooks
pre-commit install
```

### Building/Packaging
```bash
# Build package
python -m build
```

## Architecture

### Base Classes (`abcd_tools/base.py`)

The package defines an abstract base class:
- `AbstractDataset`: Interface for loading and processing datasets

Concrete dataset implementations should inherit from this base class.

### Module Organization

**`abcd_tools/task/`**
- `behavior.py`: ePrime file processing for fMRI tasks
  - `eprimeDataSet`: Loads ePrime files with task-specific column selection
  - `eprimeProcessor`: Processes MID, SST, and nBack task events (alignment, timing conversion)
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

### Configuration System

Task-specific configurations are stored in `conf/` directory as YAML files:
- `task.yaml`: Defines ePrime columns to load for each task (MID, nBack, SST)
- `dprime.yaml`: Specifies behavioral metrics for d-prime computation
- `deap_dict.yaml`, `nda_dict.yaml`: Data dictionaries for ABCD instruments
- `mappings.yaml`: Custom variable mappings

The `config_loader` utility loads these configurations at runtime, enabling flexible column selection and task-specific processing without hardcoding.

### Task Processing Pattern

Processing ePrime behavioral files follows this pattern:

1. **Load**: `eprimeDataSet.load()` reads tab-delimited ePrime files
2. **Configure**: Uses `task.yaml` to select task-specific columns
3. **Process**: `eprimeProcessor.process()` applies task-specific transformations:
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

## Testing

- Tests use pytest with coverage reporting
- Coverage configured to omit `tests/*` directory
- CI uploads coverage to Codecov
- Test files follow `test_*.py` naming convention
- Current test coverage: **53%** across core modules
- **Test Suite**: 41 passing tests covering:
  - `config_loader`: YAML I/O operations
  - `behavior`: ePrime file parsing and task-specific processing (MID, nBack, SST)
  - `metrics`: D-prime calculation components
  - `preprocess`: Image preprocessing, outlier removal, normalization, beta averaging, residualization

## Current Limitations

- **Documentation**: MkDocs structure exists but content needs completion
- **Test Data**: Some tests (d-prime computation) require actual ABCD data structure and are skipped
