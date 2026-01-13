# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

abcd-tools is a Python package for working with ABCD Study (Adolescent Brain Cognitive Development) data. It provides utilities for processing behavioral task data (ePrime files), fMRI imaging data, and computing task-based metrics like d-prime for the n-back task.

## Development Setup

```bash
# Clone and setup
git clone git@github.com:ajbarrows/abcd-tools
cd abcd-tools
mamba env update -f environment.yml
conda activate abcd-tools
python -m pip install -e .
```

## Common Development Commands

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_cli.py

# Run without coverage
pytest --no-cov
```

### Code Quality
```bash
# Format code (Black style, line length 88)
black abcd_tools/

# Lint with Ruff
ruff check abcd_tools/
ruff check --fix abcd_tools/

# Sort imports
isort abcd_tools/

# Run pre-commit hooks manually
pre-commit run --all-files
```

### Documentation
```bash
# Build MkDocs documentation
mkdocs build

# Serve docs locally
mkdocs serve
```

## Architecture Overview

### Module Structure

The package is organized into four main functional areas:

1. **Task Behavioral Processing** (`abcd_tools/task/`)
   - `behavior.py`: Contains `eprimeDataSet` for loading ePrime task files and `eprimeProcessor` for processing MID, SST, and nBack tasks
   - `metrics.py`: Contains `DPrimeDataset` for computing n-back d-prime metrics from ABCD behavioral data
   - Configuration-driven: Uses YAML files in `conf/` to specify which columns to extract from task files

2. **Image Processing** (`abcd_tools/image/`)
   - `preprocess.py`: fMRI preprocessing utilities including outlier removal, normalization, averaging across runs, and vertex-to-ROI mapping
   - `transform.py`: Image transformation utilities
   - Key functions: `compute_average_roi_betas()`, `map_hemisphere()`, `LinearResidualizer` class

3. **Download** (`abcd_tools/download/`)
   - Base classes for downloading ABCD data
   - `NDAFastTrack.py`: NDA-specific download implementation

4. **Utilities** (`abcd_tools/utils/`)
   - `ConfigLoader.py`: YAML configuration file loading (`load_yaml()`, `save_yaml()`)
   - `io.py`: Data loading/saving utilities including `pd_query_parquet()`

### Design Patterns

- **Abstract Base Classes**: The codebase uses ABC pattern extensively (see `base.py`):
  - `AbstractDataset`: Base for data loading classes
  - `AbstractDownloader`, `AbstractParser`, `AbstractReorganizer`: Base classes for download module

- **Configuration-Driven**: Heavy use of YAML configuration files in `conf/`:
  - `task.yaml`: Column specifications for ePrime task files
  - `dprime.yaml`: Mapping of ABCD variable names to simplified names for d-prime computation
  - `mappings.yaml`: Session ID mappings and anatomical atlas descriptions (Destrieux)
  - `nda_dict.yaml`, `deap_dict.yaml`: Large data dictionaries

### Key Workflows

#### Processing ePrime Task Files

```python
from abcd_tools.task.behavior import eprimeDataSet, eprimeProcessor

# Load task data - automatically extracts relevant columns from YAML config
dataset = eprimeDataSet(filepath="path/to/eprime.txt")
df = dataset.load()  # Returns DataFrame with subject ID, eventname, task columns

# Process based on task type
processor = eprimeProcessor(taskname="nback")  # or "mid", "sst"
processed = processor.process(df)  # Task-specific processing (alignment, merging, etc.)
```

The ePrime processor:
- Extracts subject ID, timepoint, and task name from filename using regex
- Aligns timings relative to scan start (subtracts prep time, converts to seconds)
- Merges cue and stimulus timings for nBack
- Task-specific processing for MID, SST, nBack

#### Computing D-Prime Metrics

```python
from abcd_tools.task.metrics import DPrimeDataset

# Initialize with optional timepoint filter
dprime = DPrimeDataset(timepoints=['baseline', 'year_1'])

# Load and compute from ABCD parquet file
results = dprime.load_and_compute(
    abcd_fpath="path/to/mri_y_tfmr_nback_beh.parquet",
    return_all=False  # True to get hit rates and false alarm rates too
)
# Returns DataFrame with dprime_0back and dprime_2back columns
```

D-prime computation:
- Uses signal detection theory: `d' = Z(hit_rate) - Z(false_alarm_rate)`
- Aggregates across emotional face conditions (negative, neutral, positive, place)
- Handles missing data by computing rates where total != 0

#### fMRI Preprocessing

```python
from abcd_tools.image.preprocess import compute_average_roi_betas, map_hemisphere

# Average betas across runs weighted by degrees of freedom
avg_betas = compute_average_roi_betas(
    run1=run1_df,
    run2=run2_df,
    motion=motion_df,  # Contains DOF for each run
    rem_outliers=True,
    outlier_std_threshold=3,
    normalize=True  # Normalize by sum of absolute values
)

# Map vertex-wise data to ROIs
roi_data = map_hemisphere(
    vertices=vertex_df,
    mapping=roi_mapping_array,  # ROI index for each vertex
    labels=roi_labels,
    prefix="lh_",
    return_statistics=True  # Also return t-values and p-values
)
```

## Configuration Files

- `conf/task.yaml`: Defines which ePrime columns to extract for each task (MID, SST, nBack)
- `conf/dprime.yaml`: Complex nested mapping for d-prime calculation variables (target correct/total, correct reject/total reject for 0-back and 2-back)
- `conf/mappings.yaml`:
  - `session_map`: Maps ABCD session IDs (ses-00A, ses-01A, etc.) to human-readable names
  - `destrieux_descriptions`: Full anatomical descriptions for Destrieux atlas ROI indices

## Code Style

- **Black** formatting (line length 88, Python 3.9+)
- **Ruff** linting (select E, F, I rules)
- **isort** with Black profile
- **Google-style** docstrings
- Pre-commit hooks enforce: YAML validation, end-of-file fixing, trailing whitespace removal, notebook output stripping, pydoclint

## Important Notes

- The package uses relative paths to find configuration files (e.g., `pathlib.Path(__file__).parents[1] / "../conf/task.yaml"`)
- ePrime filenames must follow BIDS-like convention with subject ID, timepoint, and task name for automatic parsing
- When modifying dprime calculation, note the complex nested YAML structure mapping original ABCD variable names to simplified names
- The main branch is `main` - create feature branches and PRs for changes
