"""Configuration utilities for ABCD preprocessing experiments.

This module provides functions to load experiment parameters from YAML
and generate experiment grids for testing different preprocessing options.
"""

from itertools import product

import yaml


def load_config(filepath):
    """Load configuration from a YAML file.

    Parameters
    ----------
    filepath : str
        Path to YAML config file

    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def make_experiment_grid(params):
    """Create a grid of all experiment combinations from parameters.

    Parameters
    ----------
    params : dict
        Dictionary with keys 'normalize', 'detect_outliers', 'parcellation',
        'parcellation_timing', each containing a list of options

    Returns
    -------
    list of tuple
        All combinations of (normalize, detect_outliers, parcellation, parcellation_timing).
        If 'parcellation_timing' is not in params, returns 3-tuples for backwards compatibility.

    Examples
    --------
    >>> params = {
    ...     'normalize': ['none', 'before', 'after'],
    ...     'detect_outliers': ['none', 'before', 'after'],
    ...     'parcellation': ['none', 'destrieux'],
    ...     'parcellation_timing': ['before', 'after']
    ... }
    >>> grid = make_experiment_grid(params)
    >>> len(grid)
    36
    """
    # Backwards compatibility: if parcellation_timing not provided, use 3-tuple
    if "parcellation_timing" not in params:
        return list(
            product(
                params["normalize"], params["detect_outliers"], params["parcellation"]
            )
        )

    # Generate all 4-tuple combinations
    all_combinations = list(
        product(
            params["normalize"],
            params["detect_outliers"],
            params["parcellation"],
            params["parcellation_timing"],
        )
    )

    # Filter out invalid combinations where parcellation='none' but timing is specified
    # Keep these for simplicity - timing will be ignored when parcellation='none'
    return all_combinations
