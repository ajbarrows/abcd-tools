"""YAML configuration loader utilities."""

import os
import pathlib

import yaml


def load_yaml(fpath: str | os.PathLike) -> dict:
    """Load YAML configuration file."""
    fpath = pathlib.PurePath(fpath)
    with open (fpath, 'r') as file:
        conf = yaml.safe_load(file)

    return conf
