"""Top-level API for abcd_tools.image.

This module provides utilities for:
- Image preprocessing (normalization, outlier removal)
- Parcellation mapping (vertex to ROI)
- DOF-weighted run averaging
- Image transformations (Haufe transform)
"""

from abcd_tools.image.preprocess import (
    combine_runs_weighted,
    map_hemisphere,
    normalize_by_sum,
    remove_outliers,
)
from abcd_tools.image.transform import haufe_transform

__all__ = [
    # Preprocessing
    "combine_runs_weighted",
    "map_hemisphere",
    "normalize_by_sum",
    "remove_outliers",
    # Transforms
    "haufe_transform",
]
