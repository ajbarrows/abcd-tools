"""Dataset loading utilities for tabularized beta estimates.

This module provides functions to load pre-computed first-level GLM beta
estimates from neuroimaging data in MATLAB format (HDF5/v7.3), with support
for both ROI-level and vertexwise data. It handles multi-level experimental
designs with tasks, conditions, runs, and hemispheres.
"""

from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
import pickle
import scipy.io


# =============================================================================
# Private helper functions for loading raw data
# =============================================================================


def _load_h5_matrix(
    fpath: str, key: str = "measmat", transpose: bool = True
) -> pd.DataFrame:
    """Load matrix from HDF5 file and return as DataFrame.

    Parameters
    ----------
    fpath : str
        Path to HDF5 file
    key : str, optional
        Key name in HDF5 file (default: 'measmat')
    transpose : bool, optional
        Whether to transpose the matrix (default: True)

    Returns
    -------
    pd.DataFrame
        Matrix data as DataFrame
    """
    with h5py.File(fpath, "r") as f:
        matrix = np.array(f[key])
        if transpose:
            matrix = matrix.T
        return pd.DataFrame(matrix)


def _load_early_matlab(fpath: str, key: str = "volinfo") -> pd.DataFrame:
    """Load early MATLAB format file (v7 or earlier).

    Parameters
    ----------
    fpath : str
        Path to MATLAB file
    key : str, optional
        Key to extract from file (default: 'volinfo')

    Returns
    -------
    pd.DataFrame
        Volume information as single-column DataFrame
    """
    mat_data = scipy.io.loadmat(fpath, squeeze_me=True)
    return pd.DataFrame(mat_data["visitidvec"])


def _parse_vol_info(vol_info: pd.DataFrame, idx: List[int] = [1, 2]) -> pd.DataFrame:
    """Parse volume info to extract subject ID and timepoint.

    Splits the first column on underscores and extracts specified indices
    to create subject_id and timepoint columns.

    Parameters
    ----------
    vol_info : pd.DataFrame
        Volume info DataFrame with identifier strings
    idx : List[int], optional
        Indices to extract after splitting (default: [1, 2])

    Returns
    -------
    pd.DataFrame
        DataFrame with 'subject_id' and 'timepoint' columns
    """
    parsed = vol_info.iloc[:, 0].str.split("_", expand=True)[idx]
    parsed.columns = ["participant_id", "session_id"]
    return parsed


def _join_index(betas: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    """Join beta estimates with index information.

    Parameters
    ----------
    betas : pd.DataFrame
        Beta estimate values
    idx : pd.DataFrame
        Index columns (e.g., subject_id, timepoint)

    Returns
    -------
    pd.DataFrame
        Beta estimates indexed by subject_id and timepoint
    """
    return pd.concat([idx, betas], axis=1).set_index(list(idx.columns))


# =============================================================================
# Public API functions
# =============================================================================


def load_betas(beta_path: str, vol_info_path: str) -> pd.DataFrame:
    """Load beta estimates from a single file with proper indexing.

    Loads beta values from HDF5 format, extracts subject/timepoint metadata
    from volume info file, and returns a properly indexed DataFrame.

    Parameters
    ----------
    beta_path : str
        Path to HDF5 file containing beta estimates
    vol_info_path : str
        Path to MATLAB file containing volume/visit information

    Returns
    -------
    pd.DataFrame
        Beta estimates indexed by (subject_id, timepoint)
    """
    return _load_h5_matrix(beta_path).pipe(
        _join_index, _load_early_matlab(vol_info_path).pipe(_parse_vol_info)
    )


def load_task(
    base_path: str,
    task: str = "nback",
    conditions: List[str] = ["0b", "2b"],
    runs: List[str] = ["r01", "r02"],
    hemis: List[str] = ["lh", "rh"],
    value: str = "beta",
) -> Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]:
    """Load all beta estimates for a task across conditions, runs, and hemispheres.

    Creates a nested dictionary structure containing beta estimate DataFrames
    for all combinations of conditions, runs, and hemispheres for a given task.

    Parameters
    ----------
    base_path : str
        Base directory path containing the .mat files
    task : str, optional
        Task name (default: 'nback')
    conditions : List[str], optional
        List of condition names (default: ['0b', '2b'])
    runs : List[str], optional
        List of run identifiers (default: ['r01', 'r02'])
    hemis : List[str], optional
        List of hemisphere identifiers (default: ['lh', 'rh'])
    value : str, optional
        Value type to load, e.g., 'beta', 'tstat' (default: 'beta')

    Returns
    -------
    Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]
        Nested dictionary with structure: data[task][condition][run][hemi] = DataFrame

    Examples
    --------
    >>> data = load_task('path/to/data/', task='nback')
    >>> df = data['nback']['0b']['r01']['lh']  # Access specific DataFrame
    """
    base_path = Path(base_path)
    volinfo_path = str(base_path / "vol_info.mat")

    # Create nested defaultdict for automatic key creation
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Load all combinations of task, condition, run, and hemisphere
    for task_name, condition, run, hemi in product([task], conditions, runs, hemis):
        values_path = str(
            base_path / f"{task_name}_{condition}_{value}_{run}_{hemi}.mat"
        )
        values[task_name][condition][run][hemi] = load_betas(values_path, volinfo_path)

    # Convert to regular dict for pickling (lambdas can't be pickled)
    return {
        task: {
            cond: {run: dict(hemis) for run, hemis in runs.items()}
            for cond, runs in conds.items()
        }
        for task, conds in values.items()
    }


def save_task(data: Dict, output_path: str) -> None:
    """Save task data as a single serialized pickle file.

    Serializes the entire nested dictionary structure from load_task() to disk.
    This is much faster for subsequent loads than reprocessing all .mat files.

    Parameters
    ----------
    data : Dict
        Nested dictionary from load_task() with structure:
        data[task][condition][run][hemi] = DataFrame
    output_path : str
        Path where the pickle file will be saved (e.g., 'task_data.pkl')
        Parent directories will be created if they don't exist

    Examples
    --------
    >>> data = load_task('path/to/data/', task='nback')
    >>> save_task(data, 'processed/nback_data.pkl')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_saved_task(
    input_path: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]:
    """Load task data from a serialized pickle file.

    Deserializes task data previously saved with save_task(). This is significantly
    faster than reloading all original .mat files.

    Parameters
    ----------
    input_path : str
        Path to the pickle file created by save_task()

    Returns
    -------
    Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]]
        Nested dictionary with structure: data[task][condition][run][hemi] = DataFrame

    Examples
    --------
    >>> data = load_saved_task('processed/nback_data.pkl')
    >>> df = data['nback']['0b']['r01']['lh']
    """
    input_path = Path(input_path)

    with open(input_path, "rb") as f:
        return pickle.load(f)


def load_phenotypes(
    base_path: str, phenotypes: dict, idx=["participant_id", "session_id"]
):
    """Load phenotype data from parquet files.

    Parameters
    ----------
    base_path : str
        Base directory containing phenotype parquet files
    phenotypes : dict
        Dictionary mapping table names to variable dictionaries.
        Format: {table_name: {old_col: new_col, ...}}
        If variable dict is empty, all columns are loaded.
    idx : list, optional
        Index column names (default: ['participant_id', 'session_id'])

    Returns
    -------
    pd.DataFrame
        Combined phenotype data indexed by participant and session

    Examples
    --------
    >>> phenotypes = {
    ...     'abcd_mri01': {'mri_info_deviceserialnumber': 'scanner'},
    ...     'demographics': {}  # Load all columns
    ... }
    >>> df = load_phenotypes('./data/phenotypes/', phenotypes)
    """
    base_path = Path(base_path)
    df = pd.DataFrame()

    def _filter(tmp, variables: dict):
        if len(variables) == 0:
            return tmp
        else:
            return tmp.filter(items=list(variables))

    for table, variables in phenotypes.items():
        fname = table + ".parquet"
        df = pd.concat(
            [
                df,
                (
                    pd.read_parquet(base_path / fname)
                    .set_index(idx)
                    .rename(columns=variables)
                    .pipe(_filter, variables.values())
                ),
            ],
            axis=1,
        )

    return df


def map_id(df: pd.DataFrame, params: dict, idx=["participant_id", "session_id"]):
    """Map subject IDs to match ABCD conventions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with participant_id and session_id index
    params : dict
        Configuration dict containing 'session_map' for session renaming
    idx : list, optional
        Index column names (default: ['participant_id', 'session_id'])

    Returns
    -------
    pd.DataFrame
        DataFrame with remapped IDs

    Examples
    --------
    >>> params = {'session_map': {'ses-baselineYear1Arm1': 'baseline'}}
    >>> df = map_id(df, params)
    """
    return (
        df.reset_index()
        .assign(
            participant_id=lambda x: x["participant_id"].str.replace("sub-", "INV"),
            session_id=lambda x: x["session_id"].cat.rename_categories(
                params["session_map"]
            ),
        )
        .set_index(idx)
    )


def make_contrast(task, condition1, condition2):
    """Create a contrast between two conditions.

    Parameters
    ----------
    task : dict
        Task dictionary with structure: task[condition][run][hemi] = DataFrame
    condition1 : str
        First condition name
    condition2 : str
        Second condition name

    Returns
    -------
    dict
        Updated task dictionary including the new contrast

    Examples
    --------
    >>> task = make_contrast(task, '2b', '0b')
    >>> contrast_data = task['2bv0b']['r01']['lh']
    """
    contrast = f"{condition1}v{condition2}"

    task[contrast] = {"r01": {}, "r02": {}}

    for run in ["r01", "r02"]:
        for hemi in ["lh", "rh"]:
            c1 = task[condition1][run][hemi]
            c2 = task[condition2][run][hemi]

            task[contrast][run][hemi] = c1 - c2

    return task
