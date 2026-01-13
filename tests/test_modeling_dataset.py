"""Tests for abcd_tools.modeling.dataset module."""

import pickle
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.io

from abcd_tools.modeling.dataset import (
    _join_index,
    _load_early_matlab,
    _load_h5_matrix,
    _parse_vol_info,
    load_betas,
    load_phenotypes,
    load_saved_task,
    load_task,
    make_contrast,
    map_id,
    save_task,
)


class TestPrivateHelpers:
    """Test suite for private helper functions."""

    def test_load_h5_matrix(self, tmp_path):
        """Test loading matrix from HDF5 file."""
        # Create a test HDF5 file
        test_file = tmp_path / "test.h5"
        test_matrix = np.array([[1, 2, 3], [4, 5, 6]])

        with h5py.File(test_file, "w") as f:
            f.create_dataset("measmat", data=test_matrix)

        # Load with transpose (default)
        df = _load_h5_matrix(str(test_file))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)  # Transposed

        # Load without transpose
        df_no_transpose = _load_h5_matrix(str(test_file), transpose=False)
        assert df_no_transpose.shape == (2, 3)

    def test_load_early_matlab(self, tmp_path):
        """Test loading early MATLAB format file."""
        test_file = tmp_path / "test.mat"

        # Create a simple .mat file
        visitidvec = np.array(
            ["NDAR_INV12345678_baselineYear1Arm1", "NDAR_INV87654321_baselineYear1Arm1"]
        )
        scipy.io.savemat(str(test_file), {"visitidvec": visitidvec})

        # Load the file
        df = _load_early_matlab(str(test_file))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_parse_vol_info(self):
        """Test parsing volume info to extract IDs."""
        vol_info = pd.DataFrame(
            {
                0: [
                    "NDAR_INV12345678_baselineYear1Arm1",
                    "NDAR_INV87654321_baselineYear1Arm1",
                ]
            }
        )

        parsed = _parse_vol_info(vol_info, idx=[1, 2])
        assert isinstance(parsed, pd.DataFrame)
        assert "participant_id" in parsed.columns
        assert "session_id" in parsed.columns
        assert len(parsed) == 2
        assert parsed["participant_id"].iloc[0] == "INV12345678"
        assert parsed["session_id"].iloc[0] == "baselineYear1Arm1"

    def test_join_index(self):
        """Test joining betas with index."""
        betas = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["v0", "v1", "v2"])
        idx = pd.DataFrame(
            {"participant_id": ["INV001", "INV002"], "session_id": ["baseline", "baseline"]}
        )

        result = _join_index(betas, idx)
        assert isinstance(result, pd.DataFrame)
        assert result.index.names == ["participant_id", "session_id"]
        assert result.shape == (2, 3)
        assert result.iloc[0, 0] == 1


class TestLoadBetas:
    """Test suite for load_betas function."""

    def test_load_betas(self, tmp_path):
        """Test loading beta estimates with indexing."""
        # Create test HDF5 file with beta data
        # We want 2 subjects with 3 vertices each
        # Input matrix should be 3x2 (3 vertices, 2 subjects) to be transposed to 2x3
        beta_file = tmp_path / "betas.h5"
        beta_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 vertices, 2 subjects

        with h5py.File(beta_file, "w") as f:
            f.create_dataset("measmat", data=beta_matrix)

        # Create test volume info file
        vol_file = tmp_path / "vol_info.mat"
        visitidvec = np.array(
            ["NDAR_INV12345678_baselineYear1Arm1", "NDAR_INV87654321_baselineYear1Arm1"]
        )
        scipy.io.savemat(str(vol_file), {"visitidvec": visitidvec})

        # Load betas
        df = load_betas(str(beta_file), str(vol_file))

        assert isinstance(df, pd.DataFrame)
        assert df.index.names == ["participant_id", "session_id"]
        # After transpose and join with index: 2 subjects, 3 vertices
        assert df.shape == (2, 3)


class TestLoadTask:
    """Test suite for load_task function."""

    def test_load_task(self, tmp_path):
        """Test loading full task data structure."""
        # Create minimal test data
        vol_file = tmp_path / "vol_info.mat"
        visitidvec = np.array(
            ["NDAR_INV12345678_baselineYear1Arm1", "NDAR_INV87654321_baselineYear1Arm1"]
        )
        scipy.io.savemat(str(vol_file), {"visitidvec": visitidvec})

        # Create beta files for one condition, one run, both hemispheres
        for hemi in ["lh", "rh"]:
            beta_file = tmp_path / f"nback_0b_beta_r01_{hemi}.mat"
            beta_matrix = np.random.randn(10, 2)  # 10 vertices, 2 subjects
            with h5py.File(beta_file, "w") as f:
                f.create_dataset("measmat", data=beta_matrix)

        # Load task data
        data = load_task(
            str(tmp_path),
            task="nback",
            conditions=["0b"],
            runs=["r01"],
            hemis=["lh", "rh"],
        )

        # Check structure
        assert "nback" in data
        assert "0b" in data["nback"]
        assert "r01" in data["nback"]["0b"]
        assert "lh" in data["nback"]["0b"]["r01"]
        assert "rh" in data["nback"]["0b"]["r01"]

        # Check data types
        assert isinstance(data["nback"]["0b"]["r01"]["lh"], pd.DataFrame)
        assert data["nback"]["0b"]["r01"]["lh"].shape[0] == 2  # 2 subjects


class TestSaveLoadTask:
    """Test suite for save_task and load_saved_task."""

    def test_save_and_load_task(self, tmp_path):
        """Test saving and loading task data."""
        # Create simple task structure
        task_data = {
            "nback": {
                "0b": {
                    "r01": {
                        "lh": pd.DataFrame([[1, 2]], columns=["v0", "v1"]),
                        "rh": pd.DataFrame([[3, 4]], columns=["v0", "v1"]),
                    }
                }
            }
        }

        # Save task data
        save_path = tmp_path / "task_data.pkl"
        save_task(task_data, str(save_path))

        # Check file exists
        assert save_path.exists()

        # Load task data
        loaded_data = load_saved_task(str(save_path))

        # Verify structure
        assert "nback" in loaded_data
        assert "0b" in loaded_data["nback"]
        pd.testing.assert_frame_equal(
            task_data["nback"]["0b"]["r01"]["lh"],
            loaded_data["nback"]["0b"]["r01"]["lh"],
        )


class TestLoadPhenotypes:
    """Test suite for load_phenotypes function."""

    def test_load_phenotypes(self, tmp_path):
        """Test loading phenotype data from parquet files."""
        # Create test phenotype files
        demo_data = pd.DataFrame(
            {
                "participant_id": ["INV001", "INV002"],
                "session_id": ["baseline", "baseline"],
                "age": [10, 11],
                "sex": ["M", "F"],
            }
        )
        demo_data.to_parquet(tmp_path / "demographics.parquet", index=False)

        scanner_data = pd.DataFrame(
            {
                "participant_id": ["INV001", "INV002"],
                "session_id": ["baseline", "baseline"],
                "scanner_id": ["A", "B"],
            }
        )
        scanner_data.to_parquet(tmp_path / "scanner.parquet", index=False)

        # Define phenotypes to load
        phenotypes = {
            "demographics": {"age": "age", "sex": "sex"},
            "scanner": {},  # Load all columns
        }

        # Load phenotypes
        df = load_phenotypes(str(tmp_path), phenotypes)

        assert isinstance(df, pd.DataFrame)
        assert df.index.names == ["participant_id", "session_id"]
        assert "age" in df.columns
        assert "sex" in df.columns
        assert "scanner_id" in df.columns


class TestMapId:
    """Test suite for map_id function."""

    def test_map_id(self):
        """Test ID mapping."""
        # Start with IDs in the format that map_id expects (sub- prefix)
        df = pd.DataFrame(
            {"value": [1, 2]},
            index=pd.MultiIndex.from_tuples(
                [("sub-001", "ses-baseline"), ("sub-002", "ses-baseline")],
                names=["participant_id", "session_id"],
            ),
        )

        # Make session_id categorical
        df = df.reset_index()
        df["session_id"] = pd.Categorical(df["session_id"])
        df = df.set_index(["participant_id", "session_id"])

        params = {"session_map": {"ses-baseline": "baseline"}}

        result = map_id(df, params)

        # Check participant_id mapping (sub- replaced with INV)
        assert "INV001" in result.index.get_level_values("participant_id")

        # Check session_id mapping
        assert "baseline" in result.index.get_level_values("session_id")


class TestMakeContrast:
    """Test suite for make_contrast function."""

    def test_make_contrast(self):
        """Test creating contrasts between conditions."""
        task = {
            "0b": {
                "r01": {
                    "lh": pd.DataFrame([[1, 2]], columns=["v0", "v1"]),
                    "rh": pd.DataFrame([[3, 4]], columns=["v0", "v1"]),
                },
                "r02": {
                    "lh": pd.DataFrame([[5, 6]], columns=["v0", "v1"]),
                    "rh": pd.DataFrame([[7, 8]], columns=["v0", "v1"]),
                },
            },
            "2b": {
                "r01": {
                    "lh": pd.DataFrame([[2, 3]], columns=["v0", "v1"]),
                    "rh": pd.DataFrame([[4, 5]], columns=["v0", "v1"]),
                },
                "r02": {
                    "lh": pd.DataFrame([[6, 7]], columns=["v0", "v1"]),
                    "rh": pd.DataFrame([[8, 9]], columns=["v0", "v1"]),
                },
            },
        }

        result = make_contrast(task, "2b", "0b")

        # Check contrast exists
        assert "2bv0b" in result
        assert "r01" in result["2bv0b"]
        assert "lh" in result["2bv0b"]["r01"]

        # Check values
        expected_lh_r01 = pd.DataFrame([[1, 1]], columns=["v0", "v1"])
        pd.testing.assert_frame_equal(result["2bv0b"]["r01"]["lh"], expected_lh_r01)
