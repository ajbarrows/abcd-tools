"""Parse individual ePrime files.



"""

import os
import pathlib
import re

import numpy as np
import pandas as pd

from ..base import AbstractDataset
from ..utils import config_loader


class EPrimeDataset(AbstractDataset):
    """Initialize EPrimeDataset class.

    Attributes:
        filepath (str | os.PathLike): Path to ePrime file.
        cols (list, optional): User-specified columns to load. Defaults to YAML set.
        sep (str, optional): Delimeter. Defaults to '\t'.
    """
    def __init__(self, filepath: str | os.PathLike, cols: list|str='default',
    sep: str='\t'
    ):

        self.filepath = pathlib.PurePath(filepath)
        self.fname = self.filepath.name
        self.taskname = self._find_taskname()
        self.cols = cols
        self.sep = sep

    def load(self) -> pd.DataFrame:
        """Load ePrime with Python.

        Returns:
            pd.DataFrame: ePrime events file.
        """

        def _read_csv(fpath, sep, cols) -> pd.DataFrame:
            """Helper function to read file with differing arguments."""
            return pd.read_csv(fpath, sep=sep, usecols=cols, engine="python")

        if self.cols is None:
            df = _read_csv(self.filepath, self.sep, None)
        if self.cols == 'default':
            cols = self._load_columns()
            df = _read_csv(self.filepath, self.sep, lambda c: c in set(cols))
        else:
            cols = self.cols
            df = _read_csv(self.filepath, self.sep, cols)

        df = self._insert_id_vars(df)

        return df

    def _load_columns(self, fpath: str="../conf/task.yaml"):
        """Read columns for Python eprime loading.

        Args:
            fpath (str, optional): Location of YAML file specifying columns.
                Defaults to "../conf/task.yaml".

        Returns:
            list: list of column names
        """
        p = pathlib.Path(__file__).parents[1]
        fpath = pathlib.Path(fpath)

        config_path = p / fpath
        colnames = config_loader.load_yaml(config_path)
        return colnames[self.taskname]

    def _find_subjectid(self) -> str:
        """Subset subject ID from filename."""
        return self.fname[0:16]

    def _find_timepoint(self, s:str, e: str='fMRI') -> str:
        """Subset timepoint from filename."""
        return re.findall(s+'_'+"(.*)"+'_'+e, self.fname)[0]

    def _find_taskname(self, s: str='fMRI', e: str='task') -> str:
        """Subset taskname from filename."""
        return re.findall(s+"_"+"(.*)"+"_"+e, self.fname)[0]

    def _insert_id_vars(self, df) -> pd.DataFrame:
        """Modify dataframe with file attributes."""
        subjectid = self._find_subjectid()
        df.insert(0, 'src_subject_id', subjectid)
        df.insert(1, 'eventname', self._find_timepoint(subjectid))
        df.insert(2, 'task', self.taskname)

        return df


class EPrimeProcessor():
    """Initialize ePrime processor object.

    Args:
        taskname (str): One of {"MID", "SST", "nBack"}
    """
    def __init__(self, taskname: str):
        self.taskname = taskname.lower()

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process extracted data.

        Args:
            data (pd.DataFrame): Event file from ePrimeDataSet.load().

        Returns:
            pd.DataFrame: Processed events.
        """

        if self.taskname == "mid":
            processed = self.MID_process(data)
        elif self.taskname == "sst":
            processed = self.SST_process(data)
        elif self.taskname == "nback":
            processed = self.nBack_process(data)

        return processed

    def _lowercase_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return dataframe with lowercase columnnames."""
        df.columns = [c.lower() for c in df.columns]
        return df

    def nBack_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process nBack task events.

        Args:
            df (pd.DataFrame): Event file from ePrimeDataSet.load().

        Returns:
            pd.DataFrame: Processed data.
        """
        idx = [
            'src_subject_id',
            'eventname',
            'task',
            'experimentname',
            'trial_type',
            'run',
            'onsettime',
            'offsettime',
            'duration',
            'stim.rt'
        ]

        rename = {'onsettime': 'onset'}
        drop = ['offsettime']

        df = self._lowercase_columns(df)

        df = self._nback_impose_run(df)
        df = self._nBack_drop_pre_dummy_scan(df)
        df = self._nBack_align_timings(df)

        cue_procs = ['cue0backproc', 'cue2backproc']
        df.loc[df['procedure[block]'].isin(cue_procs), 'block.type'] = 'cue'
        df['trial_type'] = df['blocktype'].combine_first(df['stimtype'])

        df = self._nBack_merge_cue_stim(df)

        df['duration'] = df['offsettime'] - df['onsettime']
        df = (df[idx]
              .dropna(subset='trial_type')
              .rename(columns=rename)
              .drop(columns=drop)
        )

        return df.reset_index(drop=True)

    def _nback_impose_run(self, df: pd.DataFrame):
        """Explicitly indicate imaging run in events file."""

        switch_r1 = 'TRSyncPROC'
        switch_r2 = 'TRSyncPROCR2'

        df['run'] = np.where(df['procedure[block]'] == switch_r1, 1,
            np.where(df['procedure[block]'] == switch_r2, 2,
            np.nan
        ))
        df['run'] = df['run'].ffill()

        return df

    def _nBack_drop_pre_dummy_scan(self, df: pd.DataFrame):
        """Drop scans before indicator time."""

        prep_vars = ['getready.rttime', 'getready2.rttime']
        if prep_vars[1] in list(df):
            df[prep_vars[0]] = df[prep_vars[0]].combine_first(df[prep_vars[1]])

        df[prep_vars[0]] = df[prep_vars[0]].ffill()

        return df.dropna(subset=prep_vars[0])


    def _nBack_align_timings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align timings by subtracting preparation time and converting to seconds."""

        # subtract prep time from onsettime/offsettime and convert to seconds
        time_cols = [c for c in df.columns if 'onsettime' in c or 'offsettime' in c]
        d = {cl: lambda x,
                cl=cl: (x[cl] - x['getready.rttime']) / 1000
                for cl in time_cols
            }
        df = df.assign(**d)

        # convert reaction time to seconds
        df['stim.rt'] = df['stim.rt'] / 1000

        return df

    def _nBack_merge_cue_stim(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge nBack task onsettimes and offsettimes."""

        df = df[df['procedure[block]'] != "Fix15secPROC"]

        # add fixation and cue durations
        df['cue.onsettime'] = df['cuefix.onsettime']
        df['cue.offsettime'] = (df['cuetarget.offsettime']
                                .combine_first(df['cue2back.offsettime'])
        )

        # merge columns

        df['onsettime'] = df['cue.onsettime'].combine_first(df['stim.onsettime'])
        df['offsettime'] = df['cue.offsettime'].combine_first(df['stim.offsettime'])

        return df

    def MID_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process MID task events.

        Args:
            df (pd.DataFrame): Event file from eprimeDataset.load()

        Returns:
            pd.DataFrame: Processed events.
        """
        #TODO place in config file
        idx = [
            'src_subject_id',
            'eventname',
            'task',
            'experimentname',
            # 'experimentversion',
            'block',
            'subtrial',
            'condition'
        ]
        vars = [
            'prbacc',
            'runmoney',
            # 'prbrt',
            'overallrt',
            'accuracy'
        ]

        times = [
            'cue.onsettime',
            'cue.offsettime',
            # 'anticipation.onsettime',
            # 'anticipation.offsettime',
            'probe.onsettime',
            'probe.offsettime',
            'probeRT.onsettime',
            'probeRT.offsettime',
            'feedback.onsettime',
            'feedback.offsettime'
        ]

        drop = ['condition','offsettime']
        rename = {'block': 'run', 'onsettime': 'onset'}
        df = self._lowercase_columns(df)

        df = self._MID_drop_pre_dummy_scan(df)
        df = self._MID_align_timings(df)
        df = self._MID_compute_probeRT(df)
        df = self._MID_parse_accuracy(df)

        df = df[idx + times + vars]
        return df

        long = df.pivot_longer(
            index=idx + vars,
            names_to = ('trial_type', '.value'),
            names_sep = '.'
        )

        long = self._MID_impose_trial_type(long)
        long['duration'] = long['offsettime'] - long['onsettime']
        long = long.rename(columns=rename)

        return long.drop(columns=drop)


    def _MID_parse_accuracy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impose explicit hit/miss."""
        df['accuracy'] = np.where(df['prbacc'] == 1, 'pos', 'neg')
        return df

    def _MID_impose_trial_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse basic trial types into specific ones."""
        df['trial_type'] = np.where(
            df['trial_type'] == 'feedback',
            df['condition'] + '_' + df['accuracy'] + '_' + df['trial_type'],
            df['condition'] + '_' + df['trial_type']
            )
        return df

    def _MID_drop_pre_dummy_scan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop scans before indicator variable."""
        ready_var = 'getready.rttime'
        prep_var = 'preptime.onsettime'
        vars = [ready_var, prep_var]

        for v in vars:
            df[v] = df[v].ffill()

        return df.dropna(subset=vars)

    def _MID_align_timings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align timings by subtracting preparation time and converting to seconds."""

        # subtract prep time from onset/offset and convert to seconds
        time_cols = [c for c in df.columns if 'onset' in c or 'offset' in c]
        time_cols.remove('preptime.onsettime')
        d = {cl: lambda x, cl=cl:
             (x[cl] - x['preptime.onsettime']) / 1000 for cl in time_cols}
        df = df.assign(**d)

        if 'overallrt' in list(df):

            # convert reaction time to seconds
            df['overallrt'][df['overallrt'] == '?'] = np.nan # who knows
            df['overallrt'] = df['overallrt'].astype('float')
            df['overallrt'] = df['overallrt'] / 1000

        else:
            df['overallrt'] = np.nan

        return df

    def _MID_compute_probeRT(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Probe RT onset/offset."""

        df['probeRT.onsettime'] = df['probe.onsettime']
        df['probeRT.offsettime'] = df['probeRT.onsettime'] + df['overallrt']

        return df


    # TODO SSRT is not calculated
    # TODO figure out why onsets are off by 0.5 seconds
    def SST_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process SST events.

        Args:
            df (pd.DataFrame): Event file from ePrimeDataSet.load().

        Returns:
            pd.DataFrame: Processed events.
        """
        SS_STIMULUS_TIME = 0.02

        df = self._lowercase_columns(df)
        df = self._SST_drop_pre_dummy_scan(df)
        df = self._SST_align_timings(df)

        idx = [
            'src_subject_id',
            'eventname',
            'task',
            'experimentname',
            # 'experimentversion',
            'trial',
            'trialcode',
            'onset',
            'offset',
            'duration'
        ]

        rename = {
            'trial': 'run',
            'trialcode': 'trial_type'
        }

        df['stopsignal.offsettime'] = df['stopsignal.starttime'] + SS_STIMULUS_TIME

        df['onset'] = df['go.onsettime'].combine_first(df['stopsignal.starttime'])
        df['offset'] = df['go.offsettime'].combine_first(df['stopsignal.offsettime'])

        df['duration'] = df['offset'] - df['onset']
        df = (df[idx]
              .reset_index(drop=True)
              .rename(columns=rename)
            )
        return df


    def _SST_drop_pre_dummy_scan(self, df: pd.DataFrame):
        """Drop scans before indicator timing."""

        prep_var = 'beginfix.starttime'
        indicator_var = 'fix.rt' # recorded for every trial
        df[prep_var] = df[prep_var].ffill()

        return df.dropna(subset=[prep_var, indicator_var])

    def _SST_align_timings(self, df: pd.DataFrame):
        """Align times by subtracting preparation and converting to seconds."""

        time_stubs = ['onsettime', 'offsettime', 'starttime']
        time_cols = [c for c in df.columns for s in time_stubs if s in c]
        time_cols.remove('beginfix.starttime')

        d = {cl: lambda x, cl=cl:
             (x[cl] - x['beginfix.starttime']) / 1000 for cl in time_cols}
        df = df.assign(**d)

        dur_cols = ['fix.rt', 'go.rt', 'ssd.rt', 'stopsignal.rt']
        d_dur = {cl: lambda x, cl=cl: (x[cl]) / 1000 for cl in dur_cols}

        return df.assign(**d_dur)
