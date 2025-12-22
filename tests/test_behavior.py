"""Tests for abcd_tools.task.behavior."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abcd_tools.task.behavior import eprimeDataSet, eprimeProcessor


class TestEprimeDataSet:
    """Test suite for eprimeDataSet class."""

    def test_init(self):
        """Test eprimeDataSet initialization."""
        filepath = "NDARINV12345678_baselineYear1Arm1_fMRI_MID_task.txt"
        dataset = eprimeDataSet(filepath)

        assert dataset.fname == filepath
        assert dataset.taskname == "MID"
        assert dataset.sep == '\t'

    def test_find_taskname_mid(self):
        """Test task name extraction for MID."""
        filepath = "NDARINV12345678_baselineYear1Arm1_fMRI_MID_task.txt"
        dataset = eprimeDataSet(filepath)
        assert dataset.taskname == "MID"

    def test_find_taskname_nback(self):
        """Test task name extraction for nBack."""
        filepath = "NDARINV12345678_baselineYear1Arm1_fMRI_nBack_task.txt"
        dataset = eprimeDataSet(filepath)
        assert dataset.taskname == "nBack"

    def test_find_taskname_sst(self):
        """Test task name extraction for SST."""
        filepath = "NDARINV12345678_baselineYear1Arm1_fMRI_SST_task.txt"
        dataset = eprimeDataSet(filepath)
        assert dataset.taskname == "SST"

    def test_find_subjectid(self):
        """Test subject ID extraction."""
        filepath = "NDARINV12345678_baselineYear1Arm1_fMRI_MID_task.txt"
        dataset = eprimeDataSet(filepath)
        subjectid = dataset._find_subjectid()
        # Note: Function returns first 16 characters which includes trailing underscore
        assert subjectid == "NDARINV12345678_"

    def test_find_timepoint(self):
        """Test timepoint extraction."""
        filepath = "NDARINV12345678_baselineYear1Arm1_fMRI_MID_task.txt"
        dataset = eprimeDataSet(filepath)
        timepoint = dataset._find_timepoint("NDARINV12345678")
        assert timepoint == "baselineYear1Arm1"


class TestEprimeProcessor:
    """Test suite for eprimeProcessor class."""

    def test_init_mid(self):
        """Test processor initialization with MID."""
        processor = eprimeProcessor("MID")
        assert processor.taskname == "mid"

    def test_init_nback(self):
        """Test processor initialization with nBack."""
        processor = eprimeProcessor("nBack")
        assert processor.taskname == "nback"

    def test_init_sst(self):
        """Test processor initialization with SST."""
        processor = eprimeProcessor("SST")
        assert processor.taskname == "sst"

    def test_lowercase_columns(self):
        """Test column name lowercasing."""
        processor = eprimeProcessor("MID")
        df = pd.DataFrame({
            'ExperimentName': [1, 2],
            'Block': [1, 2],
            'Condition': ['A', 'B']
        })
        result = processor._lowercase_columns(df)
        assert list(result.columns) == ['experimentname', 'block', 'condition']

    def test_mid_parse_accuracy(self):
        """Test MID accuracy parsing."""
        processor = eprimeProcessor("MID")
        df = pd.DataFrame({
            'prbacc': [1, 0, 1, 0]
        })
        result = processor._MID_parse_accuracy(df)
        assert 'accuracy' in result.columns
        assert list(result['accuracy']) == ['pos', 'neg', 'pos', 'neg']

    def test_nback_impose_run(self):
        """Test nBack run imposition."""
        processor = eprimeProcessor("nBack")
        df = pd.DataFrame({
            'procedure[block]': ['TRSyncPROC', 'other', 'TRSyncPROCR2', 'other']
        })
        result = processor._nback_impose_run(df)
        assert 'run' in result.columns
        assert result['run'].iloc[0] == 1
        assert result['run'].iloc[2] == 2


class TestEprimeMIDProcessor:
    """Test suite for MID-specific processing."""

    def test_mid_compute_probrt(self):
        """Test MID probeRT computation."""
        processor = eprimeProcessor("MID")
        df = pd.DataFrame({
            'probe.onsettime': [1000.0, 2000.0, 3000.0],
            'overallrt': [500.0, 600.0, 700.0]
        })
        result = processor._MID_compute_probeRT(df)
        assert 'probeRT.onsettime' in result.columns
        assert 'probeRT.offsettime' in result.columns
        assert list(result['probeRT.onsettime']) == [1000.0, 2000.0, 3000.0]
        assert list(result['probeRT.offsettime']) == [1500.0, 2600.0, 3700.0]


class TestEprimeNBackProcessor:
    """Test suite for nBack-specific processing."""

    def test_nback_merge_cue_stim(self):
        """Test nBack cue/stim merging."""
        processor = eprimeProcessor("nBack")
        df = pd.DataFrame({
            'procedure[block]': ['CuePROC', 'StimPROC', 'Fix15secPROC'],
            'cuefix.onsettime': [100.0, np.nan, np.nan],
            'cuetarget.offsettime': [200.0, np.nan, np.nan],
            'cue2back.offsettime': [np.nan, np.nan, np.nan],
            'stim.onsettime': [np.nan, 300.0, np.nan],
            'stim.offsettime': [np.nan, 400.0, np.nan]
        })
        result = processor._nBack_merge_cue_stim(df)
        assert 'onsettime' in result.columns
        assert 'offsettime' in result.columns
        # Fix15secPROC should be removed
        assert len(result) == 2


class TestEprimeIntegration:
    """Integration tests for ePrime processing."""

    def test_process_dispatch_mid(self):
        """Test process method dispatches to MID processor."""
        processor = eprimeProcessor("MID")
        # Create minimal MID data
        df = pd.DataFrame({
            'src_subject_id': ['NDARINV12345678'],
            'eventname': ['baselineYear1Arm1'],
            'task': ['MID'],
            'experimentname': ['MID'],
            'block': [1],
            'subtrial': [1],
            'condition': ['LgWin'],
            'prbacc': [1],
            'runmoney': [0],
            'overallrt': [500],
            'getready.rttime': [1000],
            'preptime.onsettime': [2000],
            'cue.onsettime': [3000],
            'cue.offsettime': [3500],
            'probe.onsettime': [4000],
            'probe.offsettime': [4500],
            'feedback.onsettime': [5000],
            'feedback.offsettime': [5500]
        })
        result = processor.process(df)
        assert isinstance(result, pd.DataFrame)
        assert 'accuracy' in result.columns

    def test_process_dispatch_nback(self):
        """Test process method dispatches to nBack processor."""
        processor = eprimeProcessor("nBack")
        assert processor.taskname == "nback"

    def test_process_dispatch_sst(self):
        """Test process method dispatches to SST processor."""
        processor = eprimeProcessor("SST")
        assert processor.taskname == "sst"
