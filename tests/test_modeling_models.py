"""Tests for abcd_tools.modeling.models module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abcd_tools.modeling.models import ExperimentResults, enet_cv, run_single_experiment


class TestEnetCV:
    """Test suite for enet_cv function."""

    def test_enet_cv_basic(self):
        """Test basic ElasticNet cross-validation."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(50) * 0.1

        models, scores = enet_cv(
            X, y, n_splits=3, n_inner_folds=2, n_alphas=10, l1_ratio=0.5, random_state=42
        )

        assert isinstance(models, dict)
        assert isinstance(scores, list)
        assert len(models) == 3  # n_splits
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    def test_enet_cv_returns_fitted_models(self):
        """Test that returned models are fitted."""
        np.random.seed(42)
        X = np.random.randn(30, 5)
        y = X[:, 0] + np.random.randn(30) * 0.1

        models, scores = enet_cv(X, y, n_splits=2, n_inner_folds=2, random_state=42)

        # Check each model can predict
        for fold_name, model_dict in models.items():
            model = model_dict["model"]
            score = model_dict["score"]

            # Model should be able to predict
            y_pred = model.predict(X)
            assert len(y_pred) == len(y)
            assert isinstance(score, float)

    def test_enet_cv_deterministic(self):
        """Test that results are deterministic with fixed random_state."""
        np.random.seed(42)
        X = np.random.randn(40, 8)
        y = X[:, 0] + np.random.randn(40) * 0.1

        models1, scores1 = enet_cv(X, y, n_splits=2, random_state=42)
        models2, scores2 = enet_cv(X, y, n_splits=2, random_state=42)

        np.testing.assert_array_almost_equal(scores1, scores2)


class TestRunSingleExperiment:
    """Test suite for run_single_experiment function."""

    def test_run_single_experiment(self):
        """Test running a single experiment."""
        # Create sample data
        index = pd.MultiIndex.from_tuples(
            [("INV001", "baseline"), ("INV002", "baseline"), ("INV003", "baseline")],
            names=["participant_id", "session_id"],
        )

        data = pd.DataFrame(
            np.random.randn(3, 10), index=index, columns=[f"v{i}" for i in range(10)]
        )

        phenotype = pd.DataFrame({"age": [10, 11, 12]}, index=index)

        params = {
            "n_splits": 2,
            "n_inner_folds": 2,
            "n_alphas": 10,
            "l1_ratio": 0.5,
            "random_state": 42,
        }

        result = run_single_experiment(
            task="nback",
            condition="0b",
            experiment=("none", "none", "none"),
            outcome="age",
            data=data,
            phenotype=phenotype,
            params=params,
        )

        assert isinstance(result, dict)
        assert result["task"] == "nback"
        assert result["condition"] == "0b"
        assert result["outcome"] == "age"
        assert "scores" in result
        assert "models" in result
        assert result["n_features"] == 10
        assert result["n_samples"] == 3


class TestExperimentResults:
    """Test suite for ExperimentResults class."""

    def test_init(self):
        """Test initialization."""
        results = ExperimentResults()
        assert results.summary_records == []
        assert results.save_path is None

        results_with_path = ExperimentResults(save_path="/tmp/results")
        assert results_with_path.save_path == Path("/tmp/results")

    def test_add_result_3tuple(self):
        """Test adding result with 3-tuple experiment."""
        results = ExperimentResults()
        results.add_result(
            task="nback",
            condition="0b",
            experiment=("none", "none", "destrieux"),
            outcome="age",
            scores=[0.1, 0.2, 0.15],
            n_features=148,
            n_samples=100,
        )

        assert len(results.summary_records) == 1
        record = results.summary_records[0]

        assert record["task"] == "nback"
        assert record["condition"] == "0b"
        assert record["normalize"] == "none"
        assert record["outliers"] == "none"
        assert record["parcellation"] == "destrieux"
        assert record["parcellation_timing"] is None
        assert record["mean_score"] == pytest.approx(0.15)
        assert "fold_0_score" in record

    def test_add_result_4tuple(self):
        """Test adding result with 4-tuple experiment."""
        results = ExperimentResults()
        results.add_result(
            task="sst",
            condition="csvcg",
            experiment=("before", "after", "destrieux", "before"),
            outcome="ssrt",
            scores=[0.05, 0.10, 0.08],
            n_features=148,
            n_samples=120,
        )

        record = results.summary_records[0]
        assert record["parcellation_timing"] == "before"

    def test_to_dataframe(self):
        """Test converting to DataFrame."""
        results = ExperimentResults()
        results.add_result(
            task="nback",
            condition="0b",
            experiment=("none", "none", "none"),
            outcome="age",
            scores=[0.1, 0.2],
            n_features=100,
            n_samples=50,
        )

        df = results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert df.index.names == ["task", "condition", "outcome"]
        assert len(df) == 1

    def test_save_and_load(self, tmp_path):
        """Test saving and loading results."""
        results = ExperimentResults()
        results.add_result(
            task="nback",
            condition="0b",
            experiment=("none", "none", "none"),
            outcome="age",
            scores=[0.1, 0.2, 0.15],
            n_features=100,
            n_samples=50,
        )

        save_path = tmp_path / "results"
        results.save(str(save_path))

        # Check files exist
        assert (save_path / "experiment_summary.parquet").exists()
        assert (save_path / "experiment_summary.csv").exists()

        # Load results
        loaded = ExperimentResults.load(str(save_path))
        assert len(loaded.summary_records) == 1
        assert loaded.summary_records[0]["task"] == "nback"

    def test_save_incremental_3tuple(self, tmp_path):
        """Test incremental saving with 3-tuple experiment."""
        save_path = tmp_path / "results"
        results = ExperimentResults(save_path=str(save_path))

        # Add first result
        results.save_incremental(
            task="nback",
            condition="0b",
            experiment=("none", "none", "destrieux"),
            outcome="age",
            scores=[0.1, 0.2],
            n_features=148,
            n_samples=100,
        )

        # Check file was created
        assert (save_path / "experiment_summary.parquet").exists()

        # Add second result
        results.save_incremental(
            task="nback",
            condition="2b",
            experiment=("none", "none", "destrieux"),
            outcome="age",
            scores=[0.15, 0.25],
            n_features=148,
            n_samples=100,
        )

        # Check data was appended
        df = pd.read_parquet(save_path / "experiment_summary.parquet")
        assert len(df) == 2

    def test_save_incremental_4tuple(self, tmp_path):
        """Test incremental saving with 4-tuple experiment."""
        save_path = tmp_path / "results"
        results = ExperimentResults(save_path=str(save_path))

        results.save_incremental(
            task="sst",
            condition="csvcg",
            experiment=("before", "after", "destrieux", "after"),
            outcome="ssrt",
            scores=[0.05, 0.10],
            n_features=148,
            n_samples=120,
        )

        df = pd.read_parquet(save_path / "experiment_summary.parquet")
        assert len(df) == 1
        assert df["parcellation_timing"].iloc[0] == "after"

    def test_save_incremental_with_models(self, tmp_path):
        """Test incremental saving with model objects."""
        save_path = tmp_path / "results"
        results = ExperimentResults(save_path=str(save_path))

        # Create dummy models
        models = {
            "cv_fold_0": {"model": "dummy_model_0", "score": 0.1},
            "cv_fold_1": {"model": "dummy_model_1", "score": 0.2},
        }

        results.save_incremental(
            task="nback",
            condition="0b",
            experiment=("none", "none", "destrieux"),
            outcome="age",
            scores=[0.1, 0.2],
            n_features=148,
            n_samples=100,
            models=models,
        )

        # Check model file was created
        model_file = (
            save_path / "models" / "nback_0b_age_none_none_destrieux_models.pkl"
        )
        assert model_file.exists()

    def test_save_incremental_no_path_raises(self):
        """Test that save_incremental raises error if no path provided."""
        results = ExperimentResults()

        with pytest.raises(ValueError, match="No save path specified"):
            results.save_incremental(
                task="nback",
                condition="0b",
                experiment=("none", "none", "none"),
                outcome="age",
                scores=[0.1],
                n_features=100,
                n_samples=50,
            )

    def test_multiple_results(self):
        """Test adding multiple results."""
        results = ExperimentResults()

        # Add multiple results
        for i in range(5):
            results.add_result(
                task="nback",
                condition=f"{i}b",
                experiment=("none", "none", "none"),
                outcome="age",
                scores=[0.1 * i, 0.2 * i],
                n_features=100,
                n_samples=50,
            )

        assert len(results.summary_records) == 5
        df = results.to_dataframe()
        assert len(df) == 5
