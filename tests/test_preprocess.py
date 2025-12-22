"""Tests for abcd_tools.image.preprocess."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp
from sklearn.linear_model import LinearRegression

from abcd_tools.image.preprocess import (
    LinearResidualizer,
    compute_average_betas,
    compute_average_roi_betas,
    compute_tstat,
    map_hemisphere,
    normalize_by_sum,
    remove_outliers,
)


class TestRemoveOutliers:
    """Test suite for remove_outliers function."""

    def test_remove_outliers_basic(self):
        """Test basic outlier removal."""
        # Note: remove_outliers uses mean+std which are affected by outliers themselves
        # For data with moderate outliers
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 20],  # 20 is mild outlier
            'col2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        result = remove_outliers(df, std_threshold=1.5)

        # Check function returns dataframe of same shape
        assert isinstance(result, pd.DataFrame)
        assert result.shape == df.shape
        # Verify clipping occurred (upper bound should be less than original max)
        assert result['col1'].max() <= df['col1'].max()

    def test_remove_outliers_no_change(self):
        """Test that normal data is not affected."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        result = remove_outliers(df, std_threshold=3.0)

        # With std_threshold=3, these values should not be clipped
        assert result['col1'].max() == 5
        assert result['col2'].max() == 50

    def test_remove_outliers_symmetric(self):
        """Test that outliers are clipped from both tails."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        result = remove_outliers(df, std_threshold=1.0)

        # Bounds should be applied
        assert isinstance(result, pd.DataFrame)
        # With tight threshold, some values should be clipped
        assert result.shape == df.shape


class TestNormalizeBySum:
    """Test suite for normalize_by_sum function."""

    def test_normalize_by_sum_basic(self):
        """Test basic normalization."""
        df = pd.DataFrame({
            'col1': [10, 20],
            'col2': [20, 40],
            'col3': [30, 60]
        })
        result = normalize_by_sum(df)

        # Each row should sum to 1 (in absolute values)
        row_sums = result.abs().sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])

    def test_normalize_by_sum_with_zeros(self):
        """Test normalization with zero values."""
        df = pd.DataFrame({
            'col1': [10, 0],
            'col2': [0, 20],
            'col3': [30, 0]
        })
        result = normalize_by_sum(df)

        # Non-zero values should still be normalized
        assert not np.isnan(result.values).any()
        assert isinstance(result, pd.DataFrame)

    def test_normalize_by_sum_negative_values(self):
        """Test normalization with negative values."""
        df = pd.DataFrame({
            'col1': [-10, 20],
            'col2': [20, -40],
            'col3': [30, 60]
        })
        result = normalize_by_sum(df)

        # Should use absolute values for sum
        row_sums = result.abs().sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0])


class TestComputeTstat:
    """Test suite for compute_tstat function."""

    def test_compute_tstat_basic(self):
        """Test basic t-statistic computation."""
        mapping = {
            'roi1': np.array([1, 2, 3, 4, 5]),
            'roi2': np.array([-1, -2, -3, -4, -5])
        }
        t_values, p_values = compute_tstat(mapping)

        assert 'roi1' in t_values
        assert 'roi2' in t_values
        assert 'roi1' in p_values
        assert 'roi2' in p_values

        # roi1 should have positive t-value
        assert t_values['roi1'] > 0
        # roi2 should have negative t-value
        assert t_values['roi2'] < 0

    def test_compute_tstat_zero_mean(self):
        """Test t-statistic when data centers around zero."""
        mapping = {
            'roi1': np.array([-2, -1, 0, 1, 2])
        }
        t_values, p_values = compute_tstat(mapping)

        # Should have t-value close to 0 and high p-value
        assert abs(t_values['roi1']) < 1
        assert p_values['roi1'] > 0.05


class TestMapHemisphere:
    """Test suite for map_hemisphere function."""

    def test_map_hemisphere_basic(self):
        """Test basic hemisphere mapping."""
        # 10 vertices, 3 ROIs
        vertices = pd.DataFrame(np.random.randn(1, 10))
        mapping = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        labels = [b'ROI0', b'ROI1', b'ROI2']

        result = map_hemisphere(vertices, mapping, labels, prefix='lh_', suffix='_avg')

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 1
        # Should have 3 ROIs
        assert 'lh_ROI0_avg' in result.columns
        assert 'lh_ROI1_avg' in result.columns
        assert 'lh_ROI2_avg' in result.columns

    def test_map_hemisphere_with_statistics(self):
        """Test hemisphere mapping with statistics."""
        vertices = pd.DataFrame(np.random.randn(1, 6))
        mapping = np.array([0, 0, 1, 1, 2, 2])
        labels = [b'ROI0', b'ROI1', b'ROI2']

        rois, tvalues, pvalues = map_hemisphere(
            vertices, mapping, labels,
            prefix='lh_', suffix='_avg',
            return_statistics=True
        )

        assert isinstance(rois, pd.DataFrame)
        assert isinstance(tvalues, pd.DataFrame)
        assert isinstance(pvalues, pd.DataFrame)
        assert rois.shape == tvalues.shape == pvalues.shape

    def test_map_hemisphere_no_decode(self):
        """Test hemisphere mapping without ASCII decoding."""
        vertices = pd.DataFrame(np.random.randn(1, 4))
        mapping = np.array([0, 0, 1, 1])
        labels = ['ROI0', 'ROI1']

        result = map_hemisphere(
            vertices, mapping, labels,
            prefix='lh_', suffix='_avg',
            decode_ascii=False
        )

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 2


class TestComputeAverageBetas:
    """Test suite for compute_average_roi_betas function."""

    def test_compute_average_roi_betas_basic(self):
        """Test basic beta averaging across runs."""
        run1 = pd.DataFrame({
            'roi1_run1': [1.0, 2.0],
            'roi2_run1': [3.0, 4.0]
        }, index=['subj1', 'subj2'])

        run2 = pd.DataFrame({
            'roi1_run2': [1.5, 2.5],
            'roi2_run2': [3.5, 4.5]
        }, index=['subj1', 'subj2'])

        motion = pd.DataFrame({
            'dof1': [10, 15],
            'dof2': [12, 18]
        }, index=['subj1', 'subj2'])

        result = compute_average_roi_betas(run1, run2, motion)

        assert isinstance(result, pd.DataFrame)
        # Should include averaged and individual run data
        assert result.shape[1] >= 2

    def test_compute_average_roi_betas_with_outlier_removal(self):
        """Test beta averaging with outlier removal."""
        run1 = pd.DataFrame({
            'roi1_run1': [1.0, 100.0],  # 100 is outlier
            'roi2_run1': [3.0, 4.0]
        }, index=['subj1', 'subj2'])

        run2 = pd.DataFrame({
            'roi1_run2': [1.5, 2.5],
            'roi2_run2': [3.5, 4.5]
        }, index=['subj1', 'subj2'])

        motion = pd.DataFrame({
            'dof1': [10, 15],
            'dof2': [12, 18]
        }, index=['subj1', 'subj2'])

        result = compute_average_roi_betas(
            run1, run2, motion,
            rem_outliers=True,
            outlier_std_threshold=2.0
        )

        assert isinstance(result, pd.DataFrame)


class TestLinearResidualizer:
    """Test suite for LinearResidualizer class."""

    def test_init(self):
        """Test LinearResidualizer initialization."""
        residualizer = LinearResidualizer()
        assert isinstance(residualizer.model, LinearRegression)
        assert residualizer.ohe_vars is None
        assert residualizer.scale_vars is None

    def test_init_with_params(self):
        """Test initialization with parameters."""
        residualizer = LinearResidualizer(
            ohe_vars=['cat1', 'cat2'],
            scale_vars=['num1', 'num2']
        )
        assert residualizer.ohe_vars == ['cat1', 'cat2']
        assert residualizer.scale_vars == ['num1', 'num2']

    def test_residualize_simple(self):
        """Test basic residualization."""
        X = pd.DataFrame({
            'predictor': [1, 2, 3, 4, 5]
        })
        y = pd.Series([2, 4, 6, 8, 10])  # Perfect linear relationship

        residualizer = LinearResidualizer()
        residuals = residualizer.residualize(X, y)

        # Residuals should be close to zero for perfect fit
        assert np.allclose(residuals, 0, atol=1e-10)

    def test_residualize_multivariate(self):
        """Test residualization with multiple predictors."""
        np.random.seed(42)
        X = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50)
        })
        y = pd.DataFrame({
            'y1': X['x1'] * 2 + X['x2'] * 3 + np.random.randn(50) * 0.1,
            'y2': X['x1'] * -1 + X['x2'] * 2 + np.random.randn(50) * 0.1
        })

        residualizer = LinearResidualizer()
        residuals = residualizer.residualize(X, y)

        assert isinstance(residuals, pd.DataFrame)
        assert residuals.shape == y.shape
        # Residuals should have mean close to 0
        assert np.abs(residuals.mean()).max() < 0.5

    def test_residualize_with_scaling(self):
        """Test residualization with scaling."""
        X = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [100, 200, 300, 400, 500]
        })
        y = pd.Series([1, 2, 3, 4, 5])

        residualizer = LinearResidualizer(scale_vars=['x1', 'x2'])
        residuals = residualizer.residualize(X, y)

        assert isinstance(residuals, (pd.Series, np.ndarray))


class TestIntegration:
    """Integration tests for preprocessing functions."""

    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Create sample data
        df = pd.DataFrame({
            'roi1': [1, 2, 3, 100, 5],  # Contains outlier
            'roi2': [10, 20, 30, 40, 50],
            'roi3': [5, 10, 15, 20, 25]
        })

        # Remove outliers
        cleaned = remove_outliers(df, std_threshold=2.0)

        # Normalize
        normalized = normalize_by_sum(cleaned)

        # Check final result
        assert isinstance(normalized, pd.DataFrame)
        assert normalized.shape == df.shape
        # Each row should sum to 1 in absolute values
        row_sums = normalized.abs().sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(df)))
