"""Machine learning modeling for ABCD fMRI data.

This module provides utilities for:
- Loading beta estimates from task fMRI data
- Configuring and managing experiment grids
- Preprocessing features (normalization, outlier removal, parcellation)
- Training predictive models using ElasticNet with nested cross-validation
- Collecting and storing experiment results

The module is organized into four submodules:
- config: Experiment configuration and grid generation
- dataset: Loading beta estimates and phenotypes from HDF5/MATLAB format
- preprocessing: Feature preprocessing pipeline and workflow orchestration
- models: Model training and results collection
"""

from abcd_tools.modeling.config import (
    load_config,
    make_experiment_grid,
)
from abcd_tools.modeling.dataset import (
    load_betas,
    load_phenotypes,
    load_saved_task,
    load_task,
    make_contrast,
    map_id,
    save_task,
)
from abcd_tools.modeling.models import (
    ExperimentResults,
    enet_cv,
    run_single_experiment,
)
from abcd_tools.modeling.preprocessing import (
    combine_hemispheres,
    load_prepared_data,
    prepare_all_experiments,
    prepare_data,
    prepare_for_preprocessing,
    residualize_features,
)

__all__ = [
    # Config
    "load_config",
    "make_experiment_grid",
    # Dataset loading
    "load_betas",
    "load_phenotypes",
    "load_saved_task",
    "load_task",
    "make_contrast",
    "map_id",
    "save_task",
    # Preprocessing
    "combine_hemispheres",
    "load_prepared_data",
    "prepare_all_experiments",
    "prepare_data",
    "prepare_for_preprocessing",
    "residualize_features",
    # Models
    "ExperimentResults",
    "enet_cv",
    "run_single_experiment",
]
