"""Data preparation utilities for vertexwise beta estimates and phenotypes.

This module provides functions to extract, combine, and prepare data from the
nested task dictionary format for preprocessing and modeling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from abcd_tools.image.preprocess import (
    combine_runs_weighted,
    map_hemisphere,
    normalize_by_sum,
    remove_outliers,
)
from nilearn.datasets import fetch_atlas_surf_destrieux
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def extract_betas(
    task_data: Dict, task: str, condition: str, run: str, hemi: str
) -> pd.DataFrame:
    """Extract beta DataFrame for a specific task/condition/run/hemisphere.

    Parameters
    ----------
    task_data : dict
        Nested dictionary with structure: data[task][condition][run][hemi]
    task : str
        Task name (e.g., 'nback', 'sst')
    condition : str
        Condition name (e.g., '0b', '2b')
    run : str
        Run identifier (e.g., 'r01', 'r02')
    hemi : str
        Hemisphere ('lh' or 'rh')

    Returns
    -------
    pd.DataFrame
        Beta estimates indexed by (participant_id, session_id)
    """
    return task_data[task][condition][run][hemi]


def combine_hemispheres(
    task_data: Dict,
    task: str,
    condition: str,
    run: str,
    hemis: List[str] = ["lh", "rh"],
) -> pd.DataFrame:
    """Combine left and right hemisphere data horizontally.

    Parameters
    ----------
    task_data : dict
        Nested task dictionary
    task : str
        Task name
    condition : str
        Condition name
    run : str
        Run identifier
    hemis : list of str, optional
        Hemispheres to combine (default: ['lh', 'rh'])

    Returns
    -------
    pd.DataFrame
        Combined beta estimates with columns from both hemispheres
    """
    dfs = []
    for hemi in hemis:
        df = extract_betas(task_data, task, condition, run, hemi)
        # Add hemisphere prefix to column names to avoid conflicts
        df = df.add_prefix(f"{hemi}_")
        dfs.append(df)

    return pd.concat(dfs, axis=1)


def combine_runs(
    task_data: Dict,
    task: str,
    condition: str,
    runs: List[str],
    hemi: str,
    method: str = "mean",
) -> pd.DataFrame:
    """Combine multiple runs using specified method.

    Parameters
    ----------
    task_data : dict
        Nested task dictionary
    task : str
        Task name
    condition : str
        Condition name
    runs : list of str
        Run identifiers to combine
    hemi : str
        Hemisphere
    method : str, optional
        Combination method: 'mean', 'concat', 'first' (default: 'mean')

    Returns
    -------
    pd.DataFrame
        Combined beta estimates
    """
    dfs = [extract_betas(task_data, task, condition, run, hemi) for run in runs]

    if method == "mean":
        # Average across runs
        return pd.concat(dfs).groupby(level=[0, 1]).mean()
    elif method == "concat":
        # Concatenate as separate features
        for i, df in enumerate(dfs):
            df.columns = [f"{col}_run{i+1}" for col in df.columns]
        return pd.concat(dfs, axis=1)
    elif method == "first":
        # Use first run only
        return dfs[0]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mean', 'concat', or 'first'")


def merge_betas_phenotypes(
    betas: pd.DataFrame, phenotypes: pd.DataFrame, how: str = "inner"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge beta estimates with phenotype data based on participant/session.

    Parameters
    ----------
    betas : pd.DataFrame
        Beta estimates indexed by (participant_id, session_id)
    phenotypes : pd.DataFrame
        Phenotype data indexed by (participant_id, session_id)
    how : str, optional
        Join type: 'inner', 'left', 'outer' (default: 'inner')

    Returns
    -------
    aligned_betas : pd.DataFrame
        Beta estimates aligned with phenotypes
    aligned_phenotypes : pd.DataFrame
        Phenotypes aligned with betas

    Examples
    --------
    >>> betas_lh = task_data['nback']['0b']['r01']['lh']
    >>> phenotypes = load_phenotypes(...)
    >>> aligned_betas, aligned_pheno = merge_betas_phenotypes(betas_lh, phenotypes)
    """
    # Merge on index (participant_id, session_id)
    merged = betas.join(phenotypes, how=how, rsuffix="_pheno")

    # Split back into betas and phenotypes
    beta_cols = betas.columns
    pheno_cols = phenotypes.columns

    aligned_betas = merged[beta_cols]
    aligned_phenotypes = merged[pheno_cols]

    return aligned_betas, aligned_phenotypes


def prepare_for_preprocessing(
    betas: pd.DataFrame,
    phenotypes: Optional[pd.DataFrame] = None,
    outcome: Optional[str] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Convert DataFrames to numpy arrays for preprocessing.

    Drops rows with missing data in either betas or outcome variable.

    Parameters
    ----------
    betas : pd.DataFrame
        Beta estimates indexed by (participant_id, session_id)
    phenotypes : pd.DataFrame, optional
        Phenotype data indexed by (participant_id, session_id)
    outcome : str, optional
        Column name in phenotypes to extract as labels

    Returns
    -------
    X : np.ndarray
        Beta matrix, shape (n_subjects, n_features)
    y : np.ndarray, optional
        Outcome labels, shape (n_subjects,). Only returned if outcome is specified.

    Examples
    --------
    >>> # Without outcome
    >>> X = prepare_for_preprocessing(betas)

    >>> # With outcome
    >>> X, y = prepare_for_preprocessing(betas, phenotypes, outcome='age')
    """
    if outcome is not None and phenotypes is not None:
        # Ensure alignment
        aligned_betas, aligned_phenotypes = merge_betas_phenotypes(
            betas, phenotypes, how="inner"
        )

        # Drop rows where outcome is missing
        valid_mask = ~aligned_phenotypes[outcome].isna()
        aligned_betas = aligned_betas[valid_mask]
        aligned_phenotypes = aligned_phenotypes[valid_mask]

        # Drop rows with any missing betas
        aligned_betas = aligned_betas.dropna(axis=0, how="any")
        aligned_phenotypes = aligned_phenotypes.loc[aligned_betas.index]

        X = aligned_betas.values
        y = aligned_phenotypes[outcome].values
        return X, y

    # Drop rows with any missing betas
    betas = betas.dropna(axis=0, how="any")
    X = betas.values

    return X


def residualize_features(
    features: pd.DataFrame,
    covariates: pd.DataFrame,
    fit_intercept: bool = True,
) -> pd.DataFrame:
    """Remove covariate effects from features using OLS regression.

    For each feature column, fits: feature = β₀ + β₁·age + β₂·sex + β₃·scanner + ε
    Returns residuals (ε) as a DataFrame with same structure as input.

    Subjects with missing covariate values are automatically dropped.

    Parameters
    ----------
    features : pd.DataFrame
        Brain features indexed by (participant_id, session_id)
    covariates : pd.DataFrame
        Covariates (age, sex, scanner) indexed by (participant_id, session_id)
    fit_intercept : bool, optional
        Whether to fit intercept in OLS (default: True)

    Returns
    -------
    pd.DataFrame
        Residualized features (subjects with missing covariates excluded)

    Raises
    ------
    ValueError
        If no overlapping subjects found or no subjects remain after dropping
        missing covariate values

    Examples
    --------
    >>> features = pd.DataFrame({'roi1': [1, 2, 3], 'roi2': [4, 5, 6]},
    ...     index=pd.MultiIndex.from_tuples([('INV1', 'baseline'),
    ...                                       ('INV2', 'baseline'),
    ...                                       ('INV3', 'baseline')]))
    >>> covariates = pd.DataFrame({'age': [10, 20, 30], 'sex': [1, 2, 1],
    ...                             'scanner': ['A', 'B', 'A']},
    ...     index=features.index)
    >>> residualized = residualize_features(features, covariates)
    >>> residualized.shape == features.shape
    True
    """
    # Align on MultiIndex (inner join - only shared subjects)
    features_aligned, covariates_aligned = features.align(
        covariates, join="inner", axis=0
    )

    if len(features_aligned) == 0:
        raise ValueError("No overlapping subjects between features and covariates")

    # Drop rows with missing covariate values
    valid_covariate_mask = ~covariates_aligned.isna().any(axis=1)
    features_aligned = features_aligned[valid_covariate_mask]
    covariates_aligned = covariates_aligned[valid_covariate_mask]

    if len(features_aligned) == 0:
        raise ValueError(
            "No subjects remaining after dropping missing covariate values"
        )

    # One-hot encode categorical covariates (sex, scanner)
    X_cov = pd.get_dummies(
        covariates_aligned,
        columns=["sex", "scanner"],
        drop_first=True,  # Prevent multicollinearity
        dummy_na=False,
    ).values

    # Residualize each feature column
    # Initialize with NaN to preserve missing values
    residuals = np.full_like(features_aligned.values, np.nan)
    for i in range(features_aligned.shape[1]):
        y_feature = features_aligned.iloc[:, i].values

        # Handle NaN in features (preserve them)
        valid_mask = ~np.isnan(y_feature)
        if valid_mask.sum() == 0:
            continue

        # Fit OLS and compute residuals
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X_cov[valid_mask], y_feature[valid_mask])
        predictions = model.predict(X_cov[valid_mask])
        residuals[valid_mask, i] = y_feature[valid_mask] - predictions

    # Return as DataFrame preserving structure
    return pd.DataFrame(
        residuals, index=features_aligned.index, columns=features_aligned.columns
    )


def get_task_data_slice(
    task_data: Dict,
    task: str,
    condition: str,
    runs: Optional[List[str]] = None,
    hemis: Optional[List[str]] = None,
    combine_runs_method: str = "mean",
    combine_hemis: bool = True,
) -> pd.DataFrame:
    """Extract and combine task data with flexible options.

    Parameters
    ----------
    task_data : dict
        Nested task dictionary
    task : str
        Task name
    condition : str
        Condition name
    runs : list of str, optional
        Runs to include. If None, uses all available runs.
    hemis : list of str, optional
        Hemispheres to include. If None, uses ['lh', 'rh'].
    combine_runs_method : str, optional
        Method to combine runs: 'mean', 'concat', 'first' (default: 'mean')
    combine_hemis : bool, optional
        Whether to combine hemispheres (default: True)

    Returns
    -------
    pd.DataFrame
        Combined beta estimates

    Examples
    --------
    >>> # Get mean of all runs, both hemispheres
    >>> data = get_task_data_slice(
    ...     task_data, task='nback', condition='2b',
    ...     runs=['r01', 'r02'], combine_runs_method='mean'
    ... )

    >>> # Get single hemisphere, single run
    >>> data = get_task_data_slice(
    ...     task_data, task='nback', condition='0b',
    ...     runs=['r01'], hemis=['lh'], combine_hemis=False
    ... )
    """
    if runs is None:
        # Get all available runs for this task/condition
        runs = list(task_data[task][condition].keys())

    if hemis is None:
        hemis = ["lh", "rh"]

    if combine_hemis and len(hemis) > 1:
        # First combine runs for each hemisphere, then combine hemispheres
        hemi_dfs = []
        for hemi in hemis:
            hemi_df = combine_runs(
                task_data, task, condition, runs, hemi, combine_runs_method
            )
            hemi_df = hemi_df.add_prefix(f"{hemi}_")
            hemi_dfs.append(hemi_df)
        return pd.concat(hemi_dfs, axis=1)
    else:
        # Just combine runs for single hemisphere
        return combine_runs(
            task_data, task, condition, runs, hemis[0], combine_runs_method
        )


def create_analysis_dataset(
    task_data: Dict,
    phenotypes: pd.DataFrame,
    task: str,
    condition: str,
    outcome: str,
    runs: Optional[List[str]] = None,
    hemis: Optional[List[str]] = None,
    combine_runs_method: str = "mean",
    combine_hemis: bool = True,
    dropna: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """Create ready-to-use analysis dataset with features and labels.

    Parameters
    ----------
    task_data : dict
        Nested task dictionary
    phenotypes : pd.DataFrame
        Phenotype data
    task : str
        Task name
    condition : str
        Condition name
    outcome : str
        Column name in phenotypes for prediction target
    runs : list of str, optional
        Runs to include
    hemis : list of str, optional
        Hemispheres to include
    combine_runs_method : str, optional
        Method to combine runs (default: 'mean')
    combine_hemis : bool, optional
        Whether to combine hemispheres (default: True)
    dropna : bool, optional
        Whether to drop rows with missing values (default: True)

    Returns
    -------
    X : np.ndarray
        Feature matrix, shape (n_subjects, n_features)
    y : np.ndarray
        Outcome labels, shape (n_subjects,)
    index : pd.Index
        MultiIndex of (participant_id, session_id) for the samples

    Examples
    --------
    >>> X, y, index = create_analysis_dataset(
    ...     task_data, phenotypes,
    ...     task='nback', condition='2b', outcome='age'
    ... )
    >>> print(f"Dataset shape: {X.shape}, n_subjects: {len(y)}")
    """
    # Get beta data
    betas = get_task_data_slice(
        task_data, task, condition, runs, hemis, combine_runs_method, combine_hemis
    )

    # Merge with phenotypes
    aligned_betas, aligned_phenotypes = merge_betas_phenotypes(
        betas, phenotypes, how="inner"
    )

    # Drop missing values if requested
    if dropna:
        # Drop rows where outcome is missing
        valid_mask = ~aligned_phenotypes[outcome].isna()
        aligned_betas = aligned_betas[valid_mask]
        aligned_phenotypes = aligned_phenotypes[valid_mask]

    # Convert to numpy
    X = aligned_betas.values
    y = aligned_phenotypes[outcome].values
    index = aligned_betas.index

    return X, y, index


def filter_qc(df: pd.DataFrame, qc: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Filter dataframe based on QC flags.

    Parameters
    ----------
    df : pd.DataFrame
        Data to filter, indexed by (participant_id, session_id)
    qc : pd.DataFrame
        QC flags indexed by (participant_id, session_id)
    variable : str
        Column name in qc to use for filtering (e.g., 'nback_incl')

    Returns
    -------
    pd.DataFrame
        Filtered data containing only rows where QC flag equals 1
    """
    qc_sub = qc.query(f"{variable} == '1'")
    return df[df.index.isin(qc_sub.index)]


def load_mri(
    task_betas: Dict,
    task: str,
    condition: str,
    qc: pd.DataFrame,
    runs: List[str] = ["r01", "r02"],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MRI beta estimates for both runs with QC filtering.

    Parameters
    ----------
    task_betas : dict
        Nested dictionary with structure: data[task][condition][run][hemi]
    task : str
        Task name (e.g., 'nback', 'sst', 'mid')
    condition : str
        Condition name (e.g., '0b', '2b')
    qc : pd.DataFrame
        QC flags indexed by (participant_id, session_id)
    runs : list of str, optional
        Run identifiers to load (default: ['r01', 'r02'])

    Returns
    -------
    run1 : pd.DataFrame
        Beta estimates for first run, QC filtered
    run2 : pd.DataFrame
        Beta estimates for second run, QC filtered
    """
    qc_var = f"{task}_incl"

    def _load_run(task_betas, task, condition, run, qc, qc_var):
        return (
            combine_hemispheres(task_betas, task, condition, run)
            .pipe(filter_qc, qc, qc_var)
            .replace({0: np.nan})
        )

    r1 = _load_run(task_betas, task, condition, runs[0], qc, qc_var)
    r2 = _load_run(task_betas, task, condition, runs[1], qc, qc_var)

    return r1, r2


# map_hemisphere and combine_runs_weighted are now imported from image.preprocess
# to avoid duplication and maintain clean architectural separation


def drop_cols_then_rows(
    df: pd.DataFrame, how_col: str = "all", how_row: str = "all"
) -> pd.DataFrame:
    """Drop columns with missing data, then drop rows with missing data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    how_col : str, optional
        How to determine if column has missing data: 'all' or 'any' (default: 'all')
    how_row : str, optional
        How to determine if row has missing data: 'all' or 'any' (default: 'all')

    Returns
    -------
    pd.DataFrame
        Dataframe with missing data removed
    """
    return df.dropna(how=how_col, axis=1).dropna(how=how_row, axis=0)


def filter_timepoint(df, timepoint="baseline"):
    """Filter dataframe to specific timepoint.

    Parameters
    ----------
    df : pd.DataFrame
        Data indexed by (participant_id, session_id)
    timepoint : str, optional
        Timepoint to filter to (default: 'baseline')

    Returns
    -------
    pd.DataFrame
        Filtered data containing only specified timepoint
    """
    return (
        df.reset_index()
        .query(f"session_id == '{timepoint}'")
        .set_index(["participant_id", "session_id"])
    )


def prepare_data(
    experiment: Union[Tuple[str, str, str], Tuple[str, str, str, str]],
    run1: pd.DataFrame,
    run2: pd.DataFrame,
    motion: pd.DataFrame,
    outlier_std_threshold: float = 3,
    parcellation_map: Optional[Dict] = None,
    covariates: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Prepare data according to experiment specification.

    Applies preprocessing steps in order specified by the experiment tuple.
    Order depends on parcellation_timing:

    If parcellation_timing='before':
    1. Parcellation (apply to each run separately)
    2. Outlier removal (before combining runs, if specified)
    3. Normalization (before combining runs, if specified)
    4. Run combination via DOF-weighted averaging
    5. Outlier removal (after combining runs, if specified)
    6. Normalization (after combining runs, if specified)
    7. Covariate residualization (if covariates provided)
    8. Drop columns/rows with all missing values

    If parcellation_timing='after':
    1. Outlier removal (before combining runs, if specified)
    2. Normalization (before combining runs, if specified)
    3. Run combination via DOF-weighted averaging
    4. Outlier removal (after combining runs, if specified)
    5. Normalization (after combining runs, if specified)
    6. Parcellation (if specified)
    7. Covariate residualization (if covariates provided)
    8. Drop columns/rows with all missing values

    Parameters
    ----------
    experiment : tuple of (str, str, str) or (str, str, str, str)
        Experiment specification: (normalize, outliers, parcellation) or
        (normalize, outliers, parcellation, parcellation_timing)
        - normalize: 'none', 'before', or 'after'
        - outliers: 'none', 'before', or 'after'
        - parcellation: 'none', 'destrieux', or other atlas name
        - parcellation_timing: 'before' or 'after' (optional, defaults to 'after')
    run1 : pd.DataFrame
        Beta estimates for first run
    run2 : pd.DataFrame
        Beta estimates for second run
    motion : pd.DataFrame
        Motion parameters for each run. Should contain DOF columns (e.g., 'task_r1_dof',
        'task_r2_dof') for run combination, and optionally avgfd columns (e.g.,
        'task_r1_avgfd', 'task_r2_avgfd') for residualization. If avgfd columns are
        present, they will be averaged across runs and included as a covariate.
    outlier_std_threshold : float, optional
        Standard deviation threshold for outlier detection (default: 3)
    parcellation_map : dict, optional
        Dictionary mapping atlas names to nilearn atlas objects.
        If None, defaults to {'destrieux': fetch_atlas_surf_destrieux()}
    covariates : pd.DataFrame, optional
        Covariate data (age, sex, scanner) indexed by (participant_id, session_id).
        If provided, features are residualized to remove covariate effects using OLS.

    Returns
    -------
    pd.DataFrame
        Preprocessed data ready for modeling
    """
    if parcellation_map is None:
        parcellation_map = {"destrieux": fetch_atlas_surf_destrieux()}

    # Unpack experiment tuple (handle both 3-tuple and 4-tuple for backwards compatibility)
    if len(experiment) == 4:
        normalize, outliers, parcellation, parcellation_timing = experiment
    else:
        normalize, outliers, parcellation = experiment
        parcellation_timing = "after"  # Default to 'after' for backwards compatibility

    def apply_parcellation(data, parcellation, parcellation_map):
        """Helper to apply parcellation to data."""
        if parcellation == "none":
            return data
        parcl = parcellation_map[parcellation]
        return pd.concat(
            [
                map_hemisphere(
                    data.filter(like="lh"),
                    mapping=parcl["map_left"],
                    labels=parcl["labels"],
                    suffix=".lh",
                    multi_subject=True,  # Enable multi-subject mode for modeling
                ),
                map_hemisphere(
                    data.filter(like="rh"),
                    mapping=parcl["map_right"],
                    labels=parcl["labels"],
                    suffix=".rh",
                    multi_subject=True,  # Enable multi-subject mode for modeling
                ),
            ],
            axis=1,
        )

    # Apply parcellation BEFORE combining runs if timing='before'
    if parcellation_timing == "before" and parcellation != "none":
        run1 = apply_parcellation(run1, parcellation, parcellation_map)
        run2 = apply_parcellation(run2, parcellation, parcellation_map)

    # Before combining runs preprocessing
    if outliers == "before":
        run1 = remove_outliers(run1, outlier_std_threshold)
        run2 = remove_outliers(run2, outlier_std_threshold)

    if normalize == "before":
        run1 = normalize_by_sum(run1)
        run2 = normalize_by_sum(run2)

    # Combine runs using DOF-weighted averaging
    # Extract only DOF columns for run combination
    dof_cols = [col for col in motion.columns if "dof" in col]
    motion_dof = motion[dof_cols]
    avg_betas = combine_runs_weighted(run1, run2, motion_dof)

    # After combining runs preprocessing
    if outliers == "after":
        avg_betas = remove_outliers(avg_betas, outlier_std_threshold)

    if normalize == "after":
        avg_betas = normalize_by_sum(avg_betas)

    # Apply parcellation AFTER preprocessing if timing='after'
    if parcellation_timing == "after":
        avg_betas = apply_parcellation(avg_betas, parcellation, parcellation_map)

    # Residualize features to remove covariate effects
    if covariates is not None:
        # Extract motion parameters (avgfd) from motion DataFrame
        motion_covariates = None
        if motion is not None and len(motion.columns) > 2:
            # Identify avgfd columns (any columns that aren't DOF columns)
            avgfd_cols = [col for col in motion.columns if "avgfd" in col]

            if avgfd_cols:
                # Average the avgfd values across runs
                # This could be DOF-weighted in the future, but simple average for now
                motion_avgfd = motion[avgfd_cols].mean(axis=1)
                motion_covariates = pd.DataFrame(
                    {"avg_fd": motion_avgfd}, index=motion.index
                )

        # Combine motion covariates with other covariates
        if motion_covariates is not None:
            # Align motion covariates with avg_betas index
            motion_covariates_aligned = motion_covariates.reindex(avg_betas.index)

            # Combine with existing covariates
            combined_covariates = covariates.join(motion_covariates_aligned, how="left")
        else:
            combined_covariates = covariates

        avg_betas = residualize_features(avg_betas, combined_covariates)

    # Drop columns/rows with all missing values
    avg_betas = avg_betas.pipe(drop_cols_then_rows)

    # Drop columns that are all zeros (provide no information for prediction)
    zero_cols = avg_betas.columns[(avg_betas == 0).all()]
    if len(zero_cols) > 0:
        logger = logging.getLogger(__name__)
        logger.info(f"Dropping {len(zero_cols)} all-zero columns after preprocessing")
        avg_betas = avg_betas.drop(columns=zero_cols)

    return avg_betas


def prepare_all_experiments(
    task_betas: Dict,
    qc: pd.DataFrame,
    motion: pd.DataFrame,
    experiment_grid: List[Tuple[str, str, str]],
    save_path: str,
    tasks: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
    timepoint: Optional[str] = None,
    covariates: Optional[pd.DataFrame] = None,
    outlier_std_threshold: float = 3,
    parcellation_map: Optional[Dict] = None,
    overwrite: bool = False,
) -> None:
    """Prepare and save data for all experiment configurations.

    This function preprocesses imaging data for all combinations of tasks,
    conditions, and experiment configurations, saving each to disk.

    Parameters
    ----------
    task_betas : dict
        Nested dictionary with structure: data[task][condition][run][hemi]
    qc : pd.DataFrame
        QC flags indexed by (participant_id, session_id)
    motion : pd.DataFrame
        Motion parameters for each run, indexed by (participant_id, session_id).
        Should contain DOF columns for each task/run (e.g., 'nback_r1_dof', 'nback_r2_dof')
        and avgfd columns (e.g., 'nback_r1_avgfd', 'nback_r2_avgfd'). DOF is used for
        run combination, avgfd is used for residualization.
    experiment_grid : list of tuple
        List of experiment configurations from make_experiment_grid()
    save_path : str
        Base directory for saving prepared data
    tasks : list of str, optional
        Task names to process. If None, processes all tasks in task_betas
    conditions : list of str, optional
        Condition/contrast names to process. If None, processes all conditions for each task.
        The same condition list is applied to all tasks being processed.
    timepoint : str, optional
        Filter to specific timepoint (e.g., 'baseline', '2_year_follow_up_y_arm_1')
        If None, includes all timepoints
    covariates : pd.DataFrame, optional
        Covariate data (age, sex, scanner) for residualization
    outlier_std_threshold : float, optional
        Standard deviation threshold for outlier detection (default: 3)
    parcellation_map : dict, optional
        Dictionary mapping atlas names to nilearn atlas objects
    overwrite : bool, optional
        Whether to overwrite existing files (default: False)

    Returns
    -------
    None
        Data is saved to disk at: {save_path}/{task}/{condition}/{experiment_key}.parquet

    Examples
    --------
    >>> from abcd_tools.modeling import prepare_all_experiments, make_experiment_grid
    >>>
    >>> # Define experiment grid
    >>> params = {
    ...     'normalize': ['none', 'before', 'after'],
    ...     'detect_outliers': ['none', 'before', 'after'],
    ...     'parcellation': ['none', 'destrieux']
    ... }
    >>> grid = make_experiment_grid(params)
    >>>
    >>> # Prepare all experiments for all tasks and conditions
    >>> prepare_all_experiments(
    ...     task_betas=betas,
    ...     qc=qc_data,
    ...     motion=motion_data,
    ...     experiment_grid=grid,
    ...     save_path='./data/prepared',
    ...     timepoint='baseline',
    ...     covariates=covars
    ... )
    >>>
    >>> # Prepare only specific conditions
    >>> prepare_all_experiments(
    ...     task_betas=betas,
    ...     qc=qc_data,
    ...     motion=motion_data,
    ...     experiment_grid=grid,
    ...     save_path='./data/prepared',
    ...     tasks=['sst'],
    ...     conditions=['csvcg', 'igvcg'],
    ...     timepoint='baseline',
    ...     covariates=covars
    ... )
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if tasks is None:
        tasks = list(task_betas.keys())

    # Calculate total number of experiments accounting for condition filter
    def count_task_conditions(task):
        all_conds = list(task_betas[task].keys())
        if conditions is not None:
            return len([c for c in all_conds if c in conditions])
        return len(all_conds)

    total = sum(count_task_conditions(task) * len(experiment_grid) for task in tasks)
    completed = 0

    for task in tasks:
        all_conditions = list(task_betas[task].keys())

        # Filter conditions if specified
        if conditions is not None:
            task_conditions = [c for c in all_conditions if c in conditions]
        else:
            task_conditions = all_conditions

        for condition in task_conditions:
            # Load and QC filter imaging data
            r1, r2 = load_mri(task_betas, task, condition, qc)

            # Filter by timepoint BEFORE prepare_data to avoid processing unnecessary data
            if timepoint is not None:
                r1 = filter_timepoint(r1, timepoint)
                r2 = filter_timepoint(r2, timepoint)

            # Filter motion data for this task (both DOF and avgfd)
            motion_cols = [
                f"{task}_r1_dof",
                f"{task}_r2_dof",
                f"{task}_r1_avgfd",
                f"{task}_r2_avgfd",
            ]
            motion_task = motion.filter(items=motion_cols)

            for experiment in experiment_grid:
                # Create experiment key for filename
                # Handle both 3-tuple and 4-tuple experiments
                if len(experiment) == 4:
                    normalize, outliers, parcellation, parcellation_timing = experiment
                    exp_key = (
                        f"{normalize}_{outliers}_{parcellation}_{parcellation_timing}"
                    )
                else:
                    normalize, outliers, parcellation = experiment
                    exp_key = f"{normalize}_{outliers}_{parcellation}"

                # Define save location
                exp_path = save_path / task / condition
                exp_path.mkdir(parents=True, exist_ok=True)
                exp_file = exp_path / f"{exp_key}.parquet"

                # Skip if file exists and not overwriting
                if exp_file.exists() and not overwrite:
                    completed += 1
                    continue

                # Prepare data according to experiment specification
                data = prepare_data(
                    experiment=experiment,
                    run1=r1,
                    run2=r2,
                    motion=motion_task,
                    outlier_std_threshold=outlier_std_threshold,
                    parcellation_map=parcellation_map,
                    covariates=covariates,
                )

                # Save to parquet
                data.to_parquet(exp_file)
                completed += 1

                # Write progress marker
                progress_file = save_path / ".progress"
                with open(progress_file, "w") as f:
                    f.write(f"{completed}/{total}: {task}/{condition}/{exp_key}\n")


def load_prepared_data(
    save_path: str,
    task: str,
    condition: str,
    experiment: Union[Tuple[str, str, str], Tuple[str, str, str, str]],
) -> pd.DataFrame:
    """Load preprocessed data for a specific experiment configuration.

    Loads data from disk and drops any feature columns that are all zeros,
    as these provide no information for prediction.

    Parameters
    ----------
    save_path : str
        Base directory where prepared data was saved
    task : str
        Task name (e.g., 'nback', 'sst')
    condition : str
        Condition name (e.g., '0b', '2b')
    experiment : tuple of (str, str, str) or (str, str, str, str)
        Experiment configuration: (normalize, outliers, parcellation) or
        (normalize, outliers, parcellation, parcellation_timing)

    Returns
    -------
    pd.DataFrame
        Preprocessed data ready for modeling, with all-zero columns removed

    Examples
    --------
    >>> data = load_prepared_data(
    ...     save_path='./data/prepared',
    ...     task='nback',
    ...     condition='0b',
    ...     experiment=('none', 'none', 'destrieux', 'after')
    ... )
    >>> data.shape
    (8500, 148)
    """
    logger = logging.getLogger(__name__)

    save_path = Path(save_path)

    # Handle both 3-tuple and 4-tuple experiments
    if len(experiment) == 4:
        normalize, outliers, parcellation, parcellation_timing = experiment
        exp_key = f"{normalize}_{outliers}_{parcellation}_{parcellation_timing}"
    else:
        normalize, outliers, parcellation = experiment
        exp_key = f"{normalize}_{outliers}_{parcellation}"

    data_file = save_path / task / condition / f"{exp_key}.parquet"

    if not data_file.exists():
        raise FileNotFoundError(
            f"Prepared data not found at {data_file}. "
            f"Run prepare_all_experiments() first."
        )

    data = pd.read_parquet(data_file)

    # Drop columns that are all zeros (provide no information for prediction)
    zero_cols = data.columns[(data == 0).all()]
    if len(zero_cols) > 0:
        logger.info(
            f"Dropping {len(zero_cols)} all-zero columns from {task}/{condition}/{exp_key}"
        )
        data = data.drop(columns=zero_cols)

    return data
