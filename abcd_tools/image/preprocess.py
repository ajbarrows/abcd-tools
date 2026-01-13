"""fMRI preprocessing functions"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.linear_model import LinearRegression


def remove_outliers(df: pd.DataFrame, std_threshold: float) -> pd.DataFrame:
    """Dynamically winsorize dataframe within columns.

    Args:
        df (pd.DataFrame): Dataframe to winsorize
        std_threshold (float): Threshold for winsorization

    Returns:
        pd.DataFrame: Winsorized dataframe
    """

    means = df.mean()
    stds = df.std()

    lower_bounds = means - std_threshold * stds
    upper_bounds = means + std_threshold * stds

    return df.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

def normalize_by_sum(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe by the sum of absolute values within each row.

    Args:
        df (pd.DataFrame): Dataframe to normalize

    Returns:
        pd.DataFrame: Normalized dataframe
    """

    def sum_nonzero_abs(row):
        # Get absolute values
        abs_vals = np.abs(row)
        # Select only non-zero values and sum them
        return np.sum(abs_vals[abs_vals != 0])

    # Apply the function to each row to get the divisors
    divisors = df.apply(sum_nonzero_abs, axis=1)

    # Divide each row by its corresponding divisor
    normalized_df = df.div(divisors, axis=0)

    return normalized_df


def compute_average_roi_betas(run1: pd.DataFrame, run2: pd.DataFrame,
                        motion: pd.DataFrame, rem_outliers: bool=False,
                        outlier_std_threshold: float=3,
                        normalize: bool=False ) -> pd.DataFrame:

    def _strip_runs(df: pd.DataFrame, names: list=['_run1', '_run2']) -> pd.DataFrame:
        """Remove run strings from column names."""
        tmp = df.copy()
        tmp.columns = [
            c.replace(n, '_all') for n in names for c in tmp.columns if n in c
            ]
        return tmp

    def _align(run1, run2, motion):
        """Align dataframes on index and columns."""
        motion.columns = ['run1_dof', 'run2_dof']
        run1 = _strip_runs(run1)
        run2 = _strip_runs(run2)

        run1, run2 = run1.align(run2, axis=1)
        run1, motion = run1.align(motion, axis=0)
        run2, motion = run2.align(motion, axis=0)

        return run1, run2, motion

    if remove_outliers:
        run1 = remove_outliers(run1, std_threshold=outlier_std_threshold)
        run2 = remove_outliers(run2, std_threshold=outlier_std_threshold)

    if normalize:
        run1 = normalize_by_sum(run1)
        run2 = normalize_by_sum(run2)

    run1_stripped, run2_stripped, motion = _align(run1, run2, motion)


    # multiply Beta values by degrees of freedom
    run1_weighted = run1_stripped.mul(motion['run1_dof'], axis=0)
    run2_weighted = run2_stripped.mul(motion['run2_dof'], axis=0)

    # divide sum by total degrees of freedom
    num = run1_weighted.add(run2_weighted, axis=0)
    den = motion['run1_dof'] + motion['run2_dof']
    avg = num.div(den, axis=0)

    # join with original data
    joined = pd.concat([avg, run1, run2], axis=1)

    # remove columns with all NaN values, then remove rows with any NaN values
    return joined.dropna(how='all', axis=1).dropna()



def compute_average_betas(run1: pd.DataFrame, run2: pd.DataFrame,
    vol_info: pd.DataFrame, motion: pd.DataFrame,
    name: str, release='r6', rem_outliers: bool=False, outlier_std_threshold: float=3,
    normalize: bool=False) -> pd.DataFrame:

    run1 = pd.concat([run1, vol_info], axis=1)
    run2 = pd.concat([run2, vol_info], axis=1)

    if release == 'r5':
        run1 = run1[run1['eventname'] == 'baseline_year_1_arm_1']
        run2 = run2[run2['eventname'] == 'baseline_year_1_arm_1']

        motion = motion.reset_index()
        motion = motion[motion['eventname'] == 'baseline_year_1_arm_1']
        motion = motion.set_index(['src_subject_id', 'eventname'])

    def _align(run1, run2, motion):
        """Align dataframes on index and columns."""
        motion.columns = ['run1_dof', 'run2_dof']

        run1, run2 = run1.align(run2, axis=1)
        run1, motion = run1.align(motion, axis=0)
        run2, motion = run2.align(motion, axis=0)

        return run1, run2, motion

    idx = ['src_subject_id', 'eventname']
    run1 = run1.set_index(idx)
    run2 = run2.set_index(idx)

    if rem_outliers:
        run1 = remove_outliers(run1, outlier_std_threshold)
        run2 = remove_outliers(run2, outlier_std_threshold)

    if normalize:
        run1 = normalize_by_sum(run1)
        run2 = normalize_by_sum(run2)

    run1_stripped, run2_stripped, motion = _align(run1, run2, motion)

    # Betas == 0 are not included in the average
    run1_stripped[run1_stripped == 0] = np.nan
    run2_stripped[run2_stripped == 0] = np.nan

    # multiply Beta values by degrees of freedom
    run1_weighted = run1_stripped.mul(motion['run1_dof'], axis=0)
    run2_weighted = run2_stripped.mul(motion['run2_dof'], axis=0)

    # divide sum by total degrees of freedom
    num = run1_weighted.add(run2_weighted, axis=0)
    den = motion['run1_dof'] + motion['run2_dof']
    avg = num.div(den, axis=0)

    if name is None:
        name = ''
    else:
        name = name + '_'

    avg.columns = [c.replace('tableData', name) for c in avg.columns]

   # remove columns and rows that are all missing
    return avg.dropna(how='all', axis=1).dropna(how='all', axis=0)

def compute_tstat(mapping: dict) -> dict:
     """Compute t-statistics.

     Args:
         lh_mapping (pd.DataFrame): Left hemisphere mapping.

     Returns:
         dict: T-statistics.
     """

     t_values = {}
     p_values = {}

     for roi, vertex in mapping.items():
         t, p = ttest_1samp(vertex, 0, axis=0, nan_policy='omit')
         t_values[roi] = t
         p_values[roi] = p

     return t_values, p_values


def map_hemisphere(vertices: pd.DataFrame, mapping: np.array, labels: list,
                   prefix: str=None, suffix: str=None,
                   decode_ascii: bool=True, return_statistics: bool=False,
                   multi_subject: bool=False
                   ) -> pd.DataFrame:
    """Map tabular vertexwise fMRI values to ROIs using nonzero average aggregation.

    Supports both single-subject mode (with optional statistics) and multi-subject
    batch processing mode.

    Args:
        vertices (pd.DataFrame): Tabular vertexwise data (columns are vertices).
            Single-subject: shape (1, n_vertices)
            Multi-subject: shape (n_subjects, n_vertices)
        mapping (np.array): Array of ROI indices. Must be the same length as n_vertices.
        labels (list): ROI labels for resulting averaged values.
        prefix (str, optional): Prefix added to all column names. Defaults to None.
        suffix (str, optional): Suffix added to all column names. Defaults to None.
        decode_ascii (bool, optional): Whether to decode labels from ASCII bytes.
            Defaults to True.
        return_statistics (bool, optional): Return t-statistics and p-values.
            Only valid for single-subject mode. Defaults to False.
        multi_subject (bool, optional): Enable multi-subject batch processing mode.
            When True, preserves all subject rows. When False, returns single-row
            DataFrame. Defaults to False.

    Returns:
        pd.DataFrame: Nonzero-averaged ROIs.
            Single-subject: shape (1, n_rois)
            Multi-subject: shape (n_subjects, n_rois)
        If return_statistics=True (single-subject only):
            tuple: (rois, tvalues, pvalues) DataFrames

    Raises:
        ValueError: If return_statistics=True and multi_subject=True
    """
    if return_statistics and multi_subject:
        raise ValueError("return_statistics is only supported in single-subject mode")

    # Handle None defaults for prefix/suffix
    if prefix is None:
        prefix = ""
    if suffix is None:
        suffix = ""

    # Decode labels if needed
    if decode_ascii:
        labels = [
            label.decode() if isinstance(label, bytes) else label for label in labels
        ]

    if multi_subject:
        # Multi-subject mode: preserve all subject rows
        # Get unique ROI indices
        unique_rois = np.unique(mapping)

        # Remove any prefix from column names
        vertices_clean = vertices.copy()
        vertices_clean.columns = vertices_clean.columns.str.replace(r".*_", "", regex=True)

        roi_dict = {}
        for roi_idx in unique_rois:
            # Get all vertices belonging to this ROI
            roi_mask = mapping == roi_idx
            roi_vertices = vertices_clean.loc[:, roi_mask]

            # Mask zeros and compute mean across vertices (axis=1)
            masked = np.ma.masked_equal(roi_vertices.values, 0)
            roi_dict[roi_idx] = np.nanmean(masked, axis=1)

        rois = pd.DataFrame(roi_dict, index=vertices.index)
        rois = rois.reindex(sorted(rois.columns), axis=1)

        # Adjust labels if needed (skip 'Unknown' label at index 0)
        if len(labels) > rois.shape[1]:
            labels = labels[1:]
        labels = [prefix + str(label).lower() + suffix for label in labels]
        rois.columns = labels

        return rois

    else:
        # Single-subject mode: original behavior
        map_dict = {}
        avg_dict = {}

        if isinstance(vertices, pd.DataFrame):
            vertices_arr = vertices.values
        else:
            vertices_arr = vertices

        for idx in mapping:
            indices = np.where(mapping == idx)[0]

            map_dict[idx] = vertices_arr[:, indices]
            map_dict[idx][map_dict[idx] == 0] = np.nan
            avg_dict[idx] = np.nanmean(map_dict[idx], axis=1)[0]
            map_dict[idx] = map_dict[idx][0]

        def _assemble_df(collection: dict, labels, prefix, suffix) -> pd.DataFrame:
            df = pd.DataFrame(collection, index=[0])
            df = df.reindex(sorted(df.columns), axis=1)

            if len(labels) > df.shape[1]:
                labels = labels[1:]

            labels = [prefix + str(label) + suffix for label in labels]
            df.columns = labels

            return df

        rois = _assemble_df(avg_dict, labels, prefix, suffix)

        if return_statistics:
            tvalues, pvalues = compute_tstat(map_dict)

            tvalues = _assemble_df(tvalues, labels, prefix, suffix)
            pvalues = _assemble_df(pvalues, labels, prefix, suffix)

            return rois, tvalues, pvalues
        else:
            return rois


def combine_runs_weighted(
    run1: pd.DataFrame,
    run2: pd.DataFrame,
    motion: pd.DataFrame,
) -> pd.DataFrame:
    """Combine two imaging runs using DOF-weighted averaging.

    Combines runs by weighting each by its degrees of freedom (DOF),
    which accounts for data quality after motion censoring. Vertices
    with beta == 0 are treated as missing data.

    This is the preferred function for DOF-weighted averaging as it provides
    clean separation of concerns (vs. compute_average_betas which includes
    additional preprocessing steps).

    Parameters
    ----------
    run1 : pd.DataFrame
        Beta estimates for first run, indexed by (participant_id, session_id)
    run2 : pd.DataFrame
        Beta estimates for second run, same indexing
    motion : pd.DataFrame
        Degrees of freedom for each run, columns: ['run1_dof', 'run2_dof']
        indexed by (participant_id, session_id)

    Returns
    -------
    pd.DataFrame
        DOF-weighted average of the two runs
    """

    def _align(run1, run2, motion):
        """Align dataframes on index and columns."""
        motion.columns = ["run1_dof", "run2_dof"]

        run1, run2 = run1.align(run2, axis=1)
        run1, motion = run1.align(motion, axis=0, join="inner")
        run2, motion = run2.align(motion, axis=0, join="inner")

        return run1, run2, motion

    run1_stripped, run2_stripped, motion = _align(run1, run2, motion)

    # Betas == 0 are not included in the average
    run1_stripped[run1_stripped == 0] = np.nan
    run2_stripped[run2_stripped == 0] = np.nan

    # Multiply beta values by degrees of freedom
    run1_weighted = run1_stripped.mul(motion["run1_dof"], axis=0)
    run2_weighted = run2_stripped.mul(motion["run2_dof"], axis=0)

    # Divide sum by total degrees of freedom
    num = run1_weighted.add(run2_weighted, axis=0)
    den = motion["run1_dof"] + motion["run2_dof"]
    avg = num.div(den, axis=0)

    return avg
