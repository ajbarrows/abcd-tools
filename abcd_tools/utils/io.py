"""ABCD data I/O utilities.
"""

import pandas as pd


def load_tabular(fpath: str,
                 cols: list=None,
                 timepoints: dict=None,
                 index: list=['src_subject_id', 'eventname']
                 ) -> pd.DataFrame:
    """Load tabular data file."""
    df = pd.read_csv(fpath)

    if timepoints:
        df = df[df['eventname'].isin(timepoints)]

    df.set_index(index, inplace=True)

    if isinstance(cols, dict):
        df = df[df.columns.intersection(set(cols.keys()))]
        df = df.rename(columns=cols)
    elif isinstance(cols, list):
        df = df[df.columns.intersection(set(cols))]

    return df

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path)

def apply_nda_names(df: pd.DataFrame,
                    abcd_dict_path: str="./abcd_5-1_dictionary.csv") -> pd.DataFrame:
    '''Replace new names with old.'''

    abcd_dict = pd.read_csv(abcd_dict_path)

    abcd_dict = abcd_dict.set_index('var_name')['var_name_nda']
    abcd_dict = abcd_dict.dropna().to_dict()

    return df.rename(columns=abcd_dict)

def parse_vol_info(vol_info: pd.DataFrame, idx=[2, 3]) -> pd.DataFrame:

    TPT_MAP = {
        "baseline": "baseline_year_1_arm_1",
        "2year": "2_year_follow_up_y_arm_1",
        "4year": "4_year_follow_up_y_arm_1",
        "6year": "6_year_follow_up_y_arm_1",
    }

    tmp = vol_info.iloc[:, 0].str.split("_", expand=True)[idx]
    tmp.columns = ["src_subject_id", "eventname"]
    tmp["src_subject_id"] = "NDAR_" + tmp["src_subject_id"]
    tmp["eventname"] = tmp["eventname"].map(TPT_MAP)

    return tmp

def load_dof(df_run1, df_run2):
    """Load degrees of freedom

    Args:
        df_run1 (pd.DataFrame): `mrisstr1bw01.csv` file
        df_run2 (pd.DataFrame): `mrisstr2bw01.csv` file

    Returns:
        pd.DataFrame: degrees of freedom for each run
    """
    idx = ['src_subject_id', 'eventname']
    df_run1 = df_run1.set_index(idx).filter(like='dof')
    df_run2 = df_run2.set_index(idx).filter(like='dof')
    dof = pd.concat([df_run1, df_run2], axis=1)
    dof.columns = ['run1_dof', 'run2_dof']

    return dof

def parse_variable_mapping(variables: dict, nda_mapping: dict,
                        deap_mapping: dict) -> dict:
    """Parse variable mapping from old releaes to new release.

    Args:
        variables (dict): dictionary of variables to be mapped
        nda_mapping (dict): mapping from old to new NDA variable names
        deap_mapping (dict): mapping from old to new DEAP variable names

    Returns:
        dict: dictionary of variables with new names
    """

    for key, value in variables.items():
        if isinstance(value, list):
            # turn list into dictionary
            variables[key] = {k: v for k, v in zip(value, value)}

    # turn into one big dictionary, make sure values are strings
    variables = {k: str(v) for d in variables.values() for k, v in d.items()}

    # ferret through various mappings
    new_variables = {}
    no_match = {}
    for key, value in variables.items():
        if key in nda_mapping.values():
            new_key = [k for k, v in nda_mapping.items() if v == key][0]
            new_variables[new_key] = value
        elif key in deap_mapping.values():
            new_key = [k for k, v in deap_mapping.items() if v == key][0]
            new_variables[new_key] = value
        elif key in nda_mapping.keys() or key in deap_mapping.keys():
            new_variables[key] = value
        else:
            # give up
            no_match[key] = value

    # need to sort this out before continuing
    ERROR = '\n'.join([f"{k} --> {v}" for k, v in no_match.items()])
    assert len(no_match) == 0, f"Some variables will not map:\n {ERROR}"

    # check to see if there are "__l" versions of the variablers
    # if so, add them to the mapping

    l_variables = {}
    for key, value in new_variables.items():
        trial = key + "__l"
        if trial in nda_mapping.keys():
            l_variables[trial] = value + '__l'

    new_variables.update(l_variables)

    return new_variables


def pd_query_parquet(
    fpath: str,
    columns: dict,
    id_vars: dict = {'participant_id': 'src_subject_id',
                    'session_id': 'eventname'},
):
    """Query parquet file using pandas.

    Args:
        fpath (str): Path to the parquet file.
        columns (dict): Dictionary of columns to select and rename.
        id_vars (dict): Dictionary of ID variables to select and rename.

    Returns:
        pd.DataFrame: DataFrame with selected columns.
    """

    columns = id_vars | columns

    return (
        pd.read_parquet(fpath, columns=list(columns.keys()))
            .rename(columns=columns)
    )
