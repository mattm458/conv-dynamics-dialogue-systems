from os import path

import pandas as pd
from pandas import DataFrame


def load_caller_metadata(switchboard_dir: str) -> DataFrame:
    """
    Load the metadata associated with all Switchboard callers.

    Parameters
    ----------
    switchboard_dir : str
        The Switchboard base directory.

    Returns
    -------
    DataFrame
        A DataFrame containing caller metadata.
    """

    df = pd.read_csv(
        path.join(switchboard_dir, "tables", "caller_tab.csv"), header=None
    )
    df.columns = pd.Index(
        [
            "caller_no",
            "pin",
            "target",
            "sex",
            "birth_year",
            "dialect_area",
            "education",
            "ti",
            "payment_type",
            "amt_pd",
            "con",
            "remarks",
            "calls_deleted",
            "speaker_partition",
        ]
    )

    return df


def load_call_metadata(switchboard_dir: str) -> DataFrame:
    """
    Load the metadata associated with all Switchboard calls.

    Parameters
    ----------
    switchboard_dir : str
        The Switchboard base directory.

    Returns
    -------
    DataFrame
        A DataFrame containing call metadata.
    """

    df = pd.read_csv(
        path.join(switchboard_dir, "tables", "call_con_tab.csv"), header=None
    )

    df.columns = pd.Index(
        [
            "conversation_no",
            "conversation_side",
            "caller_no",
            "phone_number",
            "length",
            "ivi_no",
            "remarks",
            "active",
        ]
    )

    return df
