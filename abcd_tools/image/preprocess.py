"""fMRI preprocessing functions"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
                   decode_ascii: bool=True, return_statistics: bool=False
                   ) -> pd.DataFrame:
    """Map tabular vertexwise fMRI values to ROIs using nonzero average aggregation.

    Args:
        vertices (pd.DataFrame): Tabular vertexwise data (columns are vertices).
        mapping (np.array): Array of ROI indices. Must be the same length as `vertices`.
        labels (list): ROI labels for resulting averaged values.
        prefix (str, optional): Prefix added to all column names. Defaults to None.
        suffix (str, optional): Suffix added to all column names. Defaults to None.

    Returns:
        pd.DataFrame: Nonzero-averaged ROIs.
    """

    if decode_ascii:
        labels = [label.decode() for label in labels]

    map_dict = {}
    avg_dict = {}

    if isinstance(vertices, pd.DataFrame):
        vertices = vertices.values

    for idx in mapping:

        indices = np.where(mapping == idx)[0]

        map_dict[idx] = vertices[:, indices]
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



class LinearResidualizer(BaseEstimator):

    def __init__(self, model=LinearRegression(), ohe_vars: list=None,
                scale_vars: list=None
                ):
        self.model = model
        self.ohe_vars = ohe_vars
        self.scale_vars = scale_vars

    def _transform(self, X):

        transformers = []
        if self.ohe_vars:
            ohe = OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist')
            transformers.append(('ohe', ohe, self.ohe_vars))
        if self.scale_vars:
            scaler = StandardScaler()
            transformers.append(('scale', scaler, self.scale_vars))

        preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        return preprocessor.fit_transform(X)

    def _fit(self, X, y):
        self.model.fit(X, y)
        return self

    def _predict(self, X):
        return self.model.predict(X)

    def _get_residual(self, X, y):
        return y - self._fit(X, y)._predict(X)

    def residualize(self, X, y):
        X = self._transform(X)

        if y.ndim > 1:
            residuals = np.zeros_like(y)
            for i in range(y.shape[1]):
                residuals[:, i] = self._get_residual(X, y.iloc[:, i])
            # convert back to df
            residuals = pd.DataFrame(residuals, columns=y.columns, index=y.index)
        else:
            residuals = self._get_residual(X, y)
        return residuals
