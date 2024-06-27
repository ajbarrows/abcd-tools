"""Selectively download ABCD Study FastTrack data from NDA.

This is a wrapper around `nda-tools`, but allows the user to choose
which subjects, time points, and data types to download.

Typical usage example:

    subjects=['NDARINVxxxxxxxx']
    tasks=['MID']
    timepoints=['baselineYear1Arm1']

    manifest_downloader = ManifestDownloader(dp=data_package, username=username)
    manifest_downloader.download()

    parser = ManifestParser(data_package, subjects, tasks, timepoints)
    parser.parse()
    parser.write_s3_links()

    downloader = Downloader(dp=data_package, username=username)
    downloader.download()

"""

import os
import shutil
import subprocess
import tarfile
import traceback
from pathlib import PurePath
from typing import Generator

import pandas as pd

from ..base import AbstractDownloader, AbstractParser

# TODO make sure this works for all types of data

class Downloader(AbstractDownloader):
    """Downloader object to interface with `nda-tools` `downloadcmd`.

    Attributes:
        dp (str, optional): NDA Data Package ID.
        username (str, optional): NDA account username. Defaults to None.
        download_directory (str, os.PathLike): Path to download location.
            Defaults to "."
    """
    def __init__(   # noqa: DOC301
            self, dp: str, username: str=None,
            download_directory: str | os.PathLike="."
    ):
        self.dp = dp
        self.username = username
        self.download_directory = download_directory

    def download(self, s3_links: str | os.PathLike="s3_links.csv"):
        """Make `subprocess` call to `nda-tools` `downloadcmd`.

        Args:
            s3_links (str | os.PathLike): Path to download location.
                Defaults to "s3_links.csv"
        """

        subprocess.run([
            'downloadcmd',
            '-dp', self.dp,
            '-t', s3_links,
            '-u', self.username,
            '-d', self.download_directory
        ])


class ManifestParser(AbstractParser):
    """ `nda-tools` manifest parser object.

    Attributes:
        dp (str): NDA Data Package ID.
        subjects (list, optional): List of subject IDs (e.g., ['NDARINVxxxxxxxx']).
            Defaults to None.
        timepoints (list, optional): List of REDCap-style time points
            (e.g., ['baselineYear1Arm1']). Defaults to None.
        tasks (list, optional): List of tasks (e.g., ['MID']). Defaults to None.
        metadata_filepath (str | os.PathLike, optional): Path to user-supplied
            data manifest. Defaults to None.
    """
    def __init__(
            self, dp: str, subjects: list=None, timepoints: list=None,
            tasks: list=None,
            metadata_filepath: str | os.PathLike=None
    ):
        self.dp = dp
        self.subjects = subjects
        self.timepoints = timepoints
        self.tasks=tasks
        self.metadata_filepath = metadata_filepath
        self.metadata = None
        self.s3_links = None


    def parse(self):
        """Parse metadata and generate S3 links.
        """
        self.load_metadata()
        self._parse_metadata()
        self.get_s3_links_from_metadata()

    def load_metadata(self):
        """Loads metadata from file.

        If MetadataParser.metadata_filepath is not speecified,
        attempt to load from default location.

        Raises:
            FileNotFoundError: Could not find manifest file.
        """

        fpath = self.metadata_filepath
        if not fpath:
            base_path=os.path.expanduser("~/NDA/nda-tools/downloadcmd/packages")
            fname = f'{self.dp}/package_file_metadata_{self.dp}.txt'
            fpath = os.path.join(base_path, fname)

        if not os.path.isfile(fpath):
            msg = f"Could not find manifest file at {fpath}"
            raise FileNotFoundError(msg)

        self.metadata = pd.read_csv(fpath)


    def _parse_metadata(self):
        """Break `DOWNLOAD_ALIAS` column into meaningful chunks to query.
        """

        def _parse_task(s: pd.Series) -> pd.Series:
            """Extract fMRI task name.

            Returns:
                pd.Series: Standardized task names, if available.
            """
            rep_strings = ['ABCD-MPROC-', '-fMRI']
            for string in rep_strings:
                s = s.str.replace(string, '')
            return s

        names = ['src_subject_id', 'eventname', 'task', 'date']

        alias = self.metadata['DOWNLOAD_ALIAS'].str.split('/', expand=True)
        alias = alias[2].str.split('_', expand=True)
        alias.columns = names

        alias['task'] = _parse_task(alias['task'])
        self.metadata = pd.concat([self.metadata, alias], axis=1)

    def get_s3_links_from_metadata(self):
        """Get S3 links for requested data.

        Filter for subjects, timepoints, and tasks if supplied, otherwise
        return all.
        """

        def _clean_subjects(subjects: list) -> list:
            """Standardize list of subject IDs."""
            if subjects:
                subjects = [s.replace('_', '') for s in subjects]
                subjects = [s.replace('sub-', '') for s in subjects]
            return subjects

        filter_args = {
            'src_subject_id': _clean_subjects(self.subjects),
            'eventname': self.timepoints,
            'task': self.tasks
        }

        for k, v in filter_args.items():
            if v:
                self.metadata = self.metadata[self.metadata[k].isin(v)]

        self.s3_links = self.metadata['NDA_S3_URL']

    def write_s3_links(self, fname: str | os.PathLike):
        """Write list of S3 links to a text file.

        Args:
            fname (str | os.PathLike): Destination text file (.csv).
        """
        try:
            self.s3_links.to_csv(fname, header=False, index=False)
        except Exception:
            print(traceback.format_exc())


class ManifestDownloader(Downloader):
    """`nda-tools` downloader object.

    Attributes:
        dp (str): NDA Data Package ID.
        username (str): NDA username.
    """
    def __init__(self, dp: str, username: str=None):
        self.dp = dp
        self.username = username

    def download(self) -> None:
       """
       Make an empty query through `nda-tools` to retrieve
       manifest file.
       """
       null_fp = self._write_null_file()
       downloader = Downloader(dp=self.dp, username=self.username)
       downloader.download(s3_links=null_fp)

    # TODO remove reliance on text file
    def _write_null_file(self, dir: str | os.PathLike="."):
        """Create empty text file to be passed as `s3_links`.

        Args:
            dir (str | os.PathLike, optional): File directory. Defaults to "."

        Returns:
            filename: Directory of null file.
        """
        fname = os.path.join(dir, 'none.txt')
        with open(fname, 'w') as f:
            f.write('')
        return fname

# TODO re-compress method
# TODO top level "reorganize" should do all three
class FastTrackReorganizer():
    """Reorganizer of downloaded NDA FastTrack files.

    Attributes:
        source (str | os.PathLike): Source file directory.
        target (str | os.PathLike): Destination file directory.
    """
    def __init__(self, source: str | os.PathLike, target: str | os.PathLike):
        self.source = source
        self.target = target

    def reorganize(self):
        """Extract (and combine) compressed BIDS directories (default method).
        """
        self.reorganize_compressed()

    def reorganize_compressed(self):
        """Extract (and combine) compressed BIDS directories.
        """

        for root, dirs, files in os.walk(self.source):
            for file in files:
                fpath = os.path.join(root, file)
                if tarfile.is_tarfile(fpath):
                    with tarfile.open(fpath) as tf:
                        # see tarfile.data_filter() for details
                        tf.extractall(
                            self.target, filter=tarfile.data_filter)

    def reorganize_uncompressed(self, move=True):
        """Walk through subdirectories and move or copy them into BIDS compliance.

        Args:
            move (bool, optional): Move files to target; copy if False.
                Defaults to True.
        """
        for root, dirs, _ in os.walk(self.source):
            for dir in dirs:
                if dir.startswith('sub-'):
                    walker = os.walk(root)
                    next(walker) # skip top-level directory
                    self._walk_path(walker, root, move)

    def _walk_path(self, walker: Generator, root: str | os.PathLike, move: bool):
        """Iteratively walk through subdirectories and move or copy files.

        Args:
            walker (Generator): Generated from os.walk()
            root (str | os.PathLike): Source location.
            move (bool): Move files to target; copy if False.
        """

        for r, _, files in walker:
            newpath = r.replace(root + '/', '')
            destpath = os.path.join(self.target, newpath)

            if not os.path.exists(destpath):
                os.mkdir(destpath)

            for file in files:
                sourcepath = os.path.join(r, file)

                if move:
                    self._move(sourcepath, destpath)
                else:
                    self._copy(sourcepath, destpath)

    def _move(self, sourcepath: str | os.PathLike, destpath: str | os.PathLike):
        """Move file from sourcepath to destpath.

        Args:
            sourcepath (str | os.PathLike): File to be moved.
            destpath (str | os.PathLike): File destination directory.
        """
        try:
            shutil.move(sourcepath, destpath)
        except shutil.Error:
            path = PurePath(sourcepath)
            print(f'{os.path.join(destpath, path.name)} already exists')

    def _copy(self, sourcepath: str | os.PathLike, destpath: str | os.PathLike):
        """Copy file from sourcepath to destpath.

        Args:
            sourcepath (str | os.PathLike): File to be copied.
            destpath (str | os.PathLike): File destination directory.
        """
        shutil.copy2(sourcepath, destpath)
