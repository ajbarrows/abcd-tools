"""Compute various task metrics."""

import pathlib
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..base import AbstractDataset
from ..utils import config_loader
from ..utils.io import pd_query_parquet


class DPrimeDataset(AbstractDataset):
    """Compute nBack task D-Prime."""

    def __init__(self, columns_fpath: str="../conf/dprime.yaml",
                mappings_fpath: str="../conf/mappings.yaml",
                timepoints: list=None) -> None:

        self.columns = self._load_config(columns_fpath)
        self.sessions = self._load_config(mappings_fpath)['session_map']
        self.timepoints = timepoints

    def load_and_compute(self, abcd_fpath: str, return_all=False) -> pd.DataFrame:
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
        nback_behavioral = self.load(abcd_fpath)
        dprime = self.compute_dprime(nback_behavioral, return_all)
        return dprime

    def _load_config(self, fpath: str) -> dict:
        """Load configuration file.

        Args:
            fpath (str): YAML config filepath.

        Returns:
            dict
        """
        p = pathlib.Path(__file__).parents[1]
        fpath = pathlib.Path(fpath)

        config_path = p / fpath
        vars = config_loader.load_yaml(config_path)
        return vars

    def load(self, abcd_fpath: str) -> pd.DataFrame:

        # parse config for query (a little hacky)
        dprime_loadvars = [var for value in self.columns.values() for var in value]
        dprime_loadvars = {var: var for var in dprime_loadvars} # make dict for query

        var_map = {k: v for d in self.columns.values() for k, v in d.items()}

        if self.timepoints is not None:
            pass
        else:
            pass

        return (
            pd_query_parquet(abcd_fpath, dprime_loadvars)
                .assign(eventname=lambda x:
                    x["eventname"].cat.rename_categories(self.sessions)
                    )
                .query("eventname in @timepoints")
                .rename(columns=var_map)
                # .pipe(self._correct_missing)
                .set_index(['src_subject_id', 'eventname'])
        )


    def _correct_missing(self, df: pd.DataFrame):

        N_BLOCKS = 10

        blocks = [
            'negface',
            'neutface',
            'place',
            'posface'
        ]

        conditions = ['0back', '2back']

        for (condition, block) in product(conditions, blocks):

            stem = f'{condition}_ntotal_{block}_'

            target = stem + 'target'
            lure = stem + 'lure'
            nonlure = stem + 'nonlure'

            df[nonlure] = np.where(
                df[nonlure].isna(),
                N_BLOCKS - df[target] - df[lure],
                df[nonlure]
            )
        return df

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

        def _compute_rate(df: pd.DataFrame, n_correct: list, n_total:list) -> pd.Series:
            """Helper function for rate computation."""
            correct = df[n_correct].sum(axis=1)
            total = df[n_total].sum(axis=1)

            # return correct / total
            return np.divide(correct, total,
                            out=np.zeros_like(correct),
                            where=total != 0
                        )

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
