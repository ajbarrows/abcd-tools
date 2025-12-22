"""YAML configuration loader utilities."""

import pathlib

import yaml


def load_yaml(fpath: str) -> dict:
    """Load YAML configuration file."""
    fpath = pathlib.PurePath(fpath)
    with open (fpath, 'r') as file:
        conf = yaml.safe_load(file)

    return conf

def save_yaml(data_dict: dict, fpath: str) -> None:
    """Save the mapping to a yaml file."""
    with open(fpath, 'w') as f:
        yaml.dump(data_dict, f, default_flow_style=False)
