"""Compute various task metrics."""

import pathlib

import pandas as pd
from scipy.stats import norm

from ..base import AbstractDataset
from ..utils import ConfigLoader


class DPrimeDataset(AbstractDataset):
    """Compute nBack task D-Prime."""

    def __init__(self, columns_fpath: str="../conf/dprime.yaml") -> None:
        self.columns = self._load_columns(columns_fpath)
        pass

    def load_and_compute(self, df, return_all=False) -> pd.DataFrame:
        """Load nBack behavior metrics and compute dPrime metric.

        `norminv = scipy.stats.norm.ppf`
        d-prime = norminv(hit_rate) - norminv(false_alarm_rate)

        Args:
            df (_type_): nBack behavioral metrics dataframe.
                (e.g., `mri_y_tfmr_nback_beh.csv`)
            return_all (bool, optional): Return components used to compute dPrime.
                Defaults to False.

        Returns:
            pd.DataFrame: Resulting dataframe with computed dPrime.
        """
        nback_behavioral = self.load(df)
        dprime = self.compute_dprime(nback_behavioral, return_all)
        return dprime

    def _load_columns(self, fpath: str) -> dict:
        """Load MRI behavioral columns from configuration file.

        Args:
            fpath (str): YAML config filepath.

        Returns:
            dict: Column names.
        """

        p = pathlib.Path(__file__).parents[1]
        fpath = pathlib.Path(fpath)

        config_path = p / fpath
        vars = ConfigLoader.load_yaml(config_path)
        return vars

    def load(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load and rename nBack d-prime components.

        Args:
            df (pd.DataFrame): nBack behavioral metrics dataframe.
                (e.g., `mri_y_tfmr_nback_beh.csv`)

        Returns:
            pd.DataFrame: Metrics needed to compute nBack d-prime.
        """

        df = df.set_index(['src_subject_id', 'eventname'])

        result = pd.DataFrame()
        for group, vars in self.columns.items():
            tmp = df[vars.keys()]
            tmp = tmp.rename(columns=vars)
            result = pd.concat([result, tmp], axis=1)

        return result

    def compute_dprime(self, df: pd.DataFrame, return_all=False) -> pd.DataFrame:
        """Compute n-Back 0-back and 2-back d-prime.

        `norminv = scipy.stats.norm.ppf`
        d-prime = norminv(hit_rate) - norminv(false_alarm_rate)

        Args:
            df (pd.DataFrame): d-prime components from `DPrimeDataset.load()`
            return_all (bool, optional): Return components used to
                compute d-prime. Defaults to False.

        Returns:
            pd.DataFrame: 0-back and 2-back d-prime values
        """
        cols = self.columns

        def _compute_rate(df: pd.DataFrame, n_correct: int, n_total:int) -> float:
            """Helper function for rate computation."""
            correct = df[n_correct].sum(axis=1)
            total = df[n_total].sum(axis=1)

            return correct / total

        target_correct_0back = cols['0back_target_correct'].values()
        target_total_0back = cols['0back_target_total'].values()
        reject_correct_0back = cols['0back_correct_reject'].values()
        reject_total_0back = cols['0back_total_reject'].values()

        target_correct_2back = cols['2back_target_correct'].values()
        target_total_2back = cols['2back_target_total'].values()
        reject_correct_2back = cols['2back_correct_reject'].values()
        reject_total_2back = cols['2back_total_reject'].values()

        hitrate_0back = _compute_rate(df, target_correct_0back, target_total_0back)
        hitrate_2back = _compute_rate(df, target_correct_2back, target_total_2back)
        f_alarm_0back = 1 - _compute_rate(df, reject_correct_0back, reject_total_0back)
        f_alarm_2back = 1 - _compute_rate(df, reject_correct_2back, reject_total_2back)

        dprime_0back = norm.ppf(hitrate_0back) - norm.ppf(f_alarm_0back)
        dprime_2back = norm.ppf(hitrate_2back) - norm.ppf(f_alarm_2back)

        df['dprime_0back'] = dprime_0back
        df['dprime_2back'] = dprime_2back

        if return_all:
            return df
        else:
            return df[['dprime_0back', 'dprime_2back']]
