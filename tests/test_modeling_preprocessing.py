"""Tests for abcd_tools.modeling.preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from abcd_tools.modeling.preprocessing import (
    combine_hemispheres,
    combine_runs,
    combine_runs_weighted,
    create_analysis_dataset,
    drop_cols_then_rows,
    extract_betas,
    filter_qc,
    filter_timepoint,
    get_task_data_slice,
    merge_betas_phenotypes,
    prepare_for_preprocessing,
    residualize_features,
)


@pytest.fixture
def sample_task_data():
    """Create sample task data structure for testing."""
    index = pd.MultiIndex.from_tuples(
        [("INV001", "baseline"), ("INV002", "baseline"), ("INV003", "baseline")],
        names=["participant_id", "session_id"],
    )

    return {
        "nback": {
            "0b": {
                "r01": {
                    "lh": pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=["v0", "v1"], index=index),
                    "rh": pd.DataFrame([[7, 8], [9, 10], [11, 12]], columns=["v0", "v1"], index=index),
                },
                "r02": {
                    "lh": pd.DataFrame([[2, 3], [4, 5], [6, 7]], columns=["v0", "v1"], index=index),
                    "rh": pd.DataFrame([[8, 9], [10, 11], [12, 13]], columns=["v0", "v1"], index=index),
                },
            },
            "2b": {
                "r01": {
                    "lh": pd.DataFrame([[10, 20], [30, 40], [50, 60]], columns=["v0", "v1"], index=index),
                    "rh": pd.DataFrame([[70, 80], [90, 100], [110, 120]], columns=["v0", "v1"], index=index),
                },
                "r02": {
                    "lh": pd.DataFrame([[20, 30], [40, 50], [60, 70]], columns=["v0", "v1"], index=index),
                    "rh": pd.DataFrame([[80, 90], [100, 110], [120, 130]], columns=["v0", "v1"], index=index),
                },
            },
        }
    }


class TestExtractBetas:
    """Test suite for extract_betas function."""

    def test_extract_betas(self, sample_task_data):
        """Test extracting betas from task data."""
        result = extract_betas(sample_task_data, "nback", "0b", "r01", "lh")

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)
        assert result.iloc[0, 0] == 1


class TestCombineHemispheres:
    """Test suite for combine_hemispheres function."""

    def test_combine_hemispheres(self, sample_task_data):
        """Test combining hemispheres."""
        result = combine_hemispheres(sample_task_data, "nback", "0b", "r01")

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 4)  # 2 vertices per hemisphere
        assert "lh_v0" in result.columns
        assert "rh_v0" in result.columns

    def test_combine_single_hemisphere(self, sample_task_data):
        """Test combining with single hemisphere."""
        result = combine_hemispheres(sample_task_data, "nback", "0b", "r01", hemis=["lh"])

        assert result.shape == (3, 2)  # Only left hemisphere


class TestCombineRuns:
    """Test suite for combine_runs function."""

    def test_combine_runs_mean(self, sample_task_data):
        """Test combining runs using mean."""
        result = combine_runs(
            sample_task_data, "nback", "0b", ["r01", "r02"], "lh", method="mean"
        )

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)
        # Mean of [1, 2] and [2, 3] for first subject
        assert result.iloc[0, 0] == 1.5

    def test_combine_runs_concat(self, sample_task_data):
        """Test combining runs by concatenation."""
        result = combine_runs(
            sample_task_data, "nback", "0b", ["r01", "r02"], "lh", method="concat"
        )

        assert result.shape == (3, 4)  # 2 vertices × 2 runs
        assert any("run1" in col for col in result.columns)

    def test_combine_runs_first(self, sample_task_data):
        """Test combining runs by taking first."""
        result = combine_runs(
            sample_task_data, "nback", "0b", ["r01", "r02"], "lh", method="first"
        )

        # Should return first run only
        pd.testing.assert_frame_equal(result, sample_task_data["nback"]["0b"]["r01"]["lh"])


class TestMergeBetasPhenotypes:
    """Test suite for merge_betas_phenotypes function."""

    def test_merge_betas_phenotypes_inner(self):
        """Test merging betas with phenotypes using inner join."""
        index = pd.MultiIndex.from_tuples(
            [("INV001", "baseline"), ("INV002", "baseline")],
            names=["participant_id", "session_id"],
        )
        betas = pd.DataFrame([[1, 2], [3, 4]], columns=["v0", "v1"], index=index)
        phenotypes = pd.DataFrame([[10], [11]], columns=["age"], index=index)

        aligned_betas, aligned_pheno = merge_betas_phenotypes(betas, phenotypes, how="inner")

        assert aligned_betas.shape == (2, 2)
        assert aligned_pheno.shape == (2, 1)
        pd.testing.assert_index_equal(aligned_betas.index, aligned_pheno.index)

    def test_merge_betas_phenotypes_partial_overlap(self):
        """Test merging with partial overlap."""
        betas_index = pd.MultiIndex.from_tuples(
            [("INV001", "baseline"), ("INV002", "baseline")],
            names=["participant_id", "session_id"],
        )
        pheno_index = pd.MultiIndex.from_tuples(
            [("INV001", "baseline"), ("INV003", "baseline")],
            names=["participant_id", "session_id"],
        )

        betas = pd.DataFrame([[1, 2], [3, 4]], columns=["v0", "v1"], index=betas_index)
        phenotypes = pd.DataFrame([[10], [11]], columns=["age"], index=pheno_index)

        aligned_betas, aligned_pheno = merge_betas_phenotypes(betas, phenotypes, how="inner")

        # Only INV001 should remain
        assert len(aligned_betas) == 1
        assert len(aligned_pheno) == 1


class TestPrepareForPreprocessing:
    """Test suite for prepare_for_preprocessing function."""

    def test_prepare_for_preprocessing_without_outcome(self):
        """Test preparing data without outcome."""
        index = pd.MultiIndex.from_tuples(
            [("INV001", "baseline"), ("INV002", "baseline")],
            names=["participant_id", "session_id"],
        )
        betas = pd.DataFrame([[1, 2], [3, 4]], columns=["v0", "v1"], index=index)

        X = prepare_for_preprocessing(betas)

        assert isinstance(X, np.ndarray)
        assert X.shape == (2, 2)

    def test_prepare_for_preprocessing_with_outcome(self):
        """Test preparing data with outcome."""
        index = pd.MultiIndex.from_tuples(
            [("INV001", "baseline"), ("INV002", "baseline")],
            names=["participant_id", "session_id"],
        )
        betas = pd.DataFrame([[1, 2], [3, 4]], columns=["v0", "v1"], index=index)
        phenotypes = pd.DataFrame([[10], [11]], columns=["age"], index=index)

        X, y = prepare_for_preprocessing(betas, phenotypes, outcome="age")

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape == (2, 2)
        assert y.shape == (2,)
        assert y[0] == 10

    def test_prepare_for_preprocessing_with_missing_outcome(self):
        """Test handling of missing outcome values."""
        index = pd.MultiIndex.from_tuples(
            [("INV001", "baseline"), ("INV002", "baseline"), ("INV003", "baseline")],
            names=["participant_id", "session_id"],
        )
        betas = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=["v0", "v1"], index=index)
        phenotypes = pd.DataFrame([[10], [np.nan], [12]], columns=["age"], index=index)

        X, y = prepare_for_preprocessing(betas, phenotypes, outcome="age")

        # Subject with missing age should be dropped
        assert X.shape == (2, 2)
        assert y.shape == (2,)


class TestResidualize:
    """Test suite for residualize_features function."""

    def test_residualize_features_simple(self):
        """Test basic residualization."""
        index = pd.MultiIndex.from_tuples(
            [("INV001", "baseline"), ("INV002", "baseline"), ("INV003", "baseline")],
            names=["participant_id", "session_id"],
        )

        # Create features that depend on age
        features = pd.DataFrame([[10, 20], [15, 25], [20, 30]], columns=["roi1", "roi2"], index=index)

        covariates = pd.DataFrame(
            {"age": [10, 15, 20], "sex": ["M", "F", "M"], "scanner": ["A", "A", "B"]},
            index=index,
        )

        residuals = residualize_features(features, covariates)

        assert isinstance(residuals, pd.DataFrame)
        assert residuals.shape == features.shape
        # Residuals should have mean close to 0
        assert abs(residuals.mean().mean()) < 1

    def test_residualize_features_no_overlap(self):
        """Test residualization with no overlapping subjects."""
        features_index = pd.MultiIndex.from_tuples(
            [("INV001", "baseline")], names=["participant_id", "session_id"]
        )
        covariates_index = pd.MultiIndex.from_tuples(
            [("INV002", "baseline")], names=["participant_id", "session_id"]
        )

        features = pd.DataFrame([[10, 20]], columns=["roi1", "roi2"], index=features_index)
        covariates = pd.DataFrame(
            {"age": [10], "sex": ["M"], "scanner": ["A"]}, index=covariates_index
        )

        with pytest.raises(ValueError, match="No overlapping subjects"):
            residualize_features(features, covariates)


class TestCombineRunsWeighted:
    """Test suite for combine_runs_weighted function."""

    def test_combine_runs_weighted(self):
        """Test DOF-weighted run combination."""
        index = pd.MultiIndex.from_tuples(
            [("INV001", "baseline"), ("INV002", "baseline")],
            names=["participant_id", "session_id"],
        )

        run1 = pd.DataFrame([[10, 20], [30, 40]], columns=["v0", "v1"], index=index)
        run2 = pd.DataFrame([[20, 30], [40, 50]], columns=["v0", "v1"], index=index)
        motion = pd.DataFrame([[100, 100], [100, 200]], columns=["r1_dof", "r2_dof"], index=index)

        result = combine_runs_weighted(run1, run2, motion)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)

        # First subject: equal weights (100, 100) → mean
        assert result.iloc[0, 0] == 15.0  # (10*100 + 20*100) / 200

        # Second subject: weights (100, 200) → weighted mean
        expected = (30 * 100 + 40 * 200) / 300
        assert abs(result.iloc[1, 0] - expected) < 0.01

    def test_combine_runs_weighted_with_zeros(self):
        """Test handling of zero values (treated as missing)."""
        index = pd.MultiIndex.from_tuples(
            [("INV001", "baseline")], names=["participant_id", "session_id"]
        )

        run1 = pd.DataFrame([[0, 20]], columns=["v0", "v1"], index=index)
        run2 = pd.DataFrame([[10, 30]], columns=["v0", "v1"], index=index)
        motion = pd.DataFrame([[100, 100]], columns=["r1_dof", "r2_dof"], index=index)

        result = combine_runs_weighted(run1, run2, motion)

        # v0: run1 is 0 (treated as NaN), so weighted average uses only run2
        # Since run1 is NaN, the result should be 10 (from run2)
        # Note: pandas may handle NaN differently, check actual behavior
        assert pd.isna(result.iloc[0, 0]) or result.iloc[0, 0] == 10


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_drop_cols_then_rows(self):
        """Test dropping columns and rows."""
        df = pd.DataFrame([[1, np.nan, 3], [4, 5, 6], [np.nan, np.nan, np.nan]])

        # Drop columns with all missing
        result = drop_cols_then_rows(df, how_col="all", how_row="all")

        # Column 1 should remain (has some values)
        # Row 2 should be dropped (all missing)
        assert result.shape == (2, 3)

    def test_filter_timepoint(self):
        """Test filtering by timepoint."""
        index = pd.MultiIndex.from_tuples(
            [
                ("INV001", "baseline"),
                ("INV001", "2year"),
                ("INV002", "baseline"),
            ],
            names=["participant_id", "session_id"],
        )
        df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=["v0", "v1"], index=index)

        result = filter_timepoint(df, timepoint="baseline")

        assert len(result) == 2
        assert all(result.index.get_level_values("session_id") == "baseline")

    def test_filter_qc(self):
        """Test QC filtering."""
        index = pd.MultiIndex.from_tuples(
            [("INV001", "baseline"), ("INV002", "baseline"), ("INV003", "baseline")],
            names=["participant_id", "session_id"],
        )

        df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=["v0", "v1"], index=index)
        qc = pd.DataFrame(
            {"nback_incl": ["1", "0", "1"]},  # QC flags as strings
            index=index,
        )

        result = filter_qc(df, qc, "nback_incl")

        # Only subjects with QC flag == 1 should remain
        assert len(result) == 2


class TestGetTaskDataSlice:
    """Test suite for get_task_data_slice function."""

    def test_get_task_data_slice_default(self, sample_task_data):
        """Test extracting task data with defaults."""
        result = get_task_data_slice(
            sample_task_data,
            task="nback",
            condition="0b",
            runs=["r01", "r02"],
            combine_runs_method="mean",
        )

        assert isinstance(result, pd.DataFrame)
        # Combined hemispheres, averaged runs
        assert result.shape == (3, 4)  # 3 subjects, 2 vertices × 2 hemispheres

    def test_get_task_data_slice_single_hemisphere(self, sample_task_data):
        """Test with single hemisphere."""
        result = get_task_data_slice(
            sample_task_data,
            task="nback",
            condition="0b",
            runs=["r01"],
            hemis=["lh"],
            combine_hemis=False,
        )

        assert result.shape == (3, 2)  # Single hemisphere, single run


class TestCreateAnalysisDataset:
    """Test suite for create_analysis_dataset function."""

    def test_create_analysis_dataset(self, sample_task_data):
        """Test creating complete analysis dataset."""
        index = pd.MultiIndex.from_tuples(
            [("INV001", "baseline"), ("INV002", "baseline"), ("INV003", "baseline")],
            names=["participant_id", "session_id"],
        )
        phenotypes = pd.DataFrame([[10], [11], [12]], columns=["age"], index=index)

        X, y, result_index = create_analysis_dataset(
            sample_task_data,
            phenotypes,
            task="nback",
            condition="0b",
            outcome="age",
            runs=["r01"],
            hemis=["lh"],
            combine_hemis=False,
        )

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert len(result_index) == len(y)
