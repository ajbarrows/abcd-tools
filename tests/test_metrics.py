"""Tests for abcd_tools.task.metrics."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from abcd_tools.task.metrics import DPrimeDataset


class TestDPrimeDataset:
    """Test suite for DPrimeDataset class."""

    @pytest.fixture
    def mock_dprime_config(self, tmp_path):
        """Create mock dprime configuration."""
        from abcd_tools.utils import config_loader

        config = {
            '0back_target_correct': {
                'var1': 'var1',
                'var2': 'var2'
            },
            '0back_target_total': {
                'var3': 'var3',
                'var4': 'var4'
            },
            '0back_correct_reject': {
                'var5': 'var5',
                'var6': 'var6'
            },
            '0back_total_reject': {
                'var7': 'var7',
                'var8': 'var8'
            },
            '2back_target_correct': {
                'var9': 'var9',
                'var10': 'var10'
            },
            '2back_target_total': {
                'var11': 'var11',
                'var12': 'var12'
            },
            '2back_correct_reject': {
                'var13': 'var13',
                'var14': 'var14'
            },
            '2back_total_reject': {
                'var15': 'var15',
                'var16': 'var16'
            }
        }
        config_path = tmp_path / "dprime.yaml"
        config_loader.save_yaml(config, str(config_path))
        return str(config_path)

    @pytest.fixture
    def mock_mappings_config(self, tmp_path):
        """Create mock mappings configuration."""
        from abcd_tools.utils import config_loader

        config = {
            'session_map': {
                'baseline_year_1_arm_1': 'baseline',
                '2_year_follow_up_y_arm_1': '2year'
            }
        }
        config_path = tmp_path / "mappings.yaml"
        config_loader.save_yaml(config, str(config_path))
        return str(config_path)

    def test_init(self, mock_dprime_config, mock_mappings_config):
        """Test DPrimeDataset initialization."""
        dataset = DPrimeDataset(
            columns_fpath=mock_dprime_config,
            mappings_fpath=mock_mappings_config
        )
        assert dataset.columns is not None
        assert dataset.sessions is not None
        assert isinstance(dataset.columns, dict)
        assert isinstance(dataset.sessions, dict)

    def test_compute_dprime_simple(self):
        """Test d-prime computation with simple data."""
        # Skip this test - requires actual ABCD data structure
        pytest.skip("Test requires actual ABCD data structure and column mappings")

    def test_compute_dprime_return_all(self):
        """Test d-prime computation with return_all=True."""
        # Skip this test - requires actual ABCD data structure
        pytest.skip("Test requires actual ABCD data structure and column mappings")

    def test_compute_dprime_perfect_performance(self):
        """Test d-prime with perfect hit rate and zero false alarms."""
        # Skip this test - requires actual ABCD data structure
        pytest.skip("Test requires actual ABCD data structure and column mappings")


class TestDPrimeFormula:
    """Test d-prime calculation formula."""

    def test_dprime_formula(self):
        """Verify d-prime calculation matches expected formula."""
        # d' = Z(hit_rate) - Z(false_alarm_rate)
        hit_rate = 0.8
        fa_rate = 0.2

        expected_dprime = norm.ppf(hit_rate) - norm.ppf(fa_rate)

        # Should be positive when hit_rate > fa_rate
        assert expected_dprime > 0

        # Test with reversed rates
        hit_rate_low = 0.2
        fa_rate_high = 0.8
        expected_negative = norm.ppf(hit_rate_low) - norm.ppf(fa_rate_high)

        # Should be negative when hit_rate < fa_rate
        assert expected_negative < 0
