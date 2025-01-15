"""fMRI preprocessing functions"""
from typing import Tuple

import numpy as np
import pandas as pd


def windsorize_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """_summary_

    Args:
        df (pd.DataFrame): Tabular activation Beta weights. Columns are
            vertices/ROIs, rows are observations.

    Returns:
        Tuple[pd.DataFrame, dict]: Beta weights with outliers Windsorized, statistics.
    """

    # Calculate thresholds
    meanf = np.nanmean(df, axis=0)
    stdf = np.nanstd(df, axis=0)

    up_thresh = meanf + (3 * stdf)
    low_thresh = meanf - (3 * stdf)

    # Windsorize outliers
    outlier_mask = (df > up_thresh) | (df < low_thresh)
    outlier_count = np.sum(outlier_mask)

    # Clip outliers to the thresholds
    df_clipped = np.clip(df, low_thresh, up_thresh)

    total_non_nan = np.count_nonzero(~np.isnan(df_clipped))
    outlier_percentage = (outlier_count / total_non_nan) * 100

    stats = {
        'outlier_count': outlier_count,
        'outlier_percentage': outlier_percentage,
        'up_thresh': up_thresh,
        'low_thresh': low_thresh
    }

    return df_clipped, stats

def map_hemisphere(vertices: pd.DataFrame, mapping: np.array, labels: list,
                   prefix: str=None, suffix: str=None,
                   decode_ascii: bool=True) -> pd.DataFrame:
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

    for idx in mapping:

        vertex = vertices.values[:, idx]

        if idx in map_dict:
            joined = np.array([map_dict[idx], vertex])
            masked = np.ma.masked_equal(joined, 0) # only compute mean of nonzeros
            map_dict[idx] = np.nanmean( masked, axis=0)
        else:
            map_dict[idx] = vertex

    rois = pd.DataFrame(map_dict)
    rois = rois.reindex(sorted(rois.columns), axis=1)

    if len(labels) > rois.shape[1]:
        labels = labels[1:]

    labels = [prefix + str(label) + suffix for label in labels]
    rois.columns = labels
    return rois
