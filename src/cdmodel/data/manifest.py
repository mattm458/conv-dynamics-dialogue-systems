from os import path
from typing import Final

from pandas import DataFrame

from cdmodel.consts import FEATURES, FEATURES_NORM_BY_CONV_SPEAKER

_MANIFEST_VERSION: Final[int] = 1

_MANIFEST_SCHEMA: Final[dict] = {
    "id": "The conversation ID",
    "start": "Start time of the segment in seconds",
    "end": "End time of the segment in seconds",
    "side": "The side of the conversation, either A or B",
    "side_partner": "The side of the speaker's partner, either A or B",
    "speaker_id": "The ID of the person speaking in the segment",
    "speaker_id_partner": "The ID of the speaker's partner in the segment",
    "transcript": "A transcript of the words spoken in the segment",
    "gender": "The gender of the person speaking in the segment",
    "gender_partner": "The gender of the speaker's partner in the segment",
    "da": "A colon-separated list of dialogue acts in the segment, if available",
    "da_consolidated": "Out of all dialogue acts, in this segment, the most commonly used dialogue act in the dataset",
    "da_category": "da_consolidated simplified into broad dialogue act categories",
}
for f in FEATURES + FEATURES_NORM_BY_CONV_SPEAKER:
    _MANIFEST_SCHEMA[f] = f


class DatasetVersionError(Exception):
    def __init__(
        self,
        dataset_version: int,
        supported_version: int = _MANIFEST_VERSION,
    ):
        """
        An exception indicating a mismatch between the expected and actual dataset version

        Parameters
        ----------
        dataset_version : int
            The given version of the dataset.
        supported_version : int, optional
            The version supported by the module. The latest version by default.
        """
        message = f"Preprocessed dataset version {dataset_version} is too old. Version supported: {supported_version}"
        super().__init__(message)


def get_dataset_version(dir: str) -> int:
    """
    Get the version number of a preprocessed conversational dataset.

    Parameters
    ----------
    dir : str
        A path to the preprocessed dataset directory.

    Returns
    -------
    int
        The version number of the dataset.
    """
    manifest_path = path.join(dir, "MANIFEST")

    with open(manifest_path, "r") as infile:
        return int(infile.readline())


def write_manifest(out_dir: str) -> None:
    """
    Write the manifest to the dataset output directory. Alternatively, if the manifest is already there, check whether
    the current version of the dataset matches the version in the manifest.

    Parameters
    ----------
    out_dir : str
        A path to the directory where the manifest should be writen.

    Raises
    ------
    DatasetVersionError
        Raised if a manifest already exists in out_dir, and its version is not the same as the version
        currently being used for preprocessing output.
    """
    manifest_path = path.join(out_dir, "MANIFEST")

    if path.exists(manifest_path):
        version = get_dataset_version(out_dir)
        if version != _MANIFEST_VERSION:
            raise DatasetVersionError(dataset_version=version)
    else:
        with open(manifest_path, "w") as outfile:
            outfile.write(str(_MANIFEST_VERSION))


def validate_df(df: DataFrame) -> None:
    """
    Validates a DataFrame according to the preprocessing schema.

    Parameters
    ----------
    df : DataFrame
        A DataFrame to validate

    Raises
    ------
    Exception
        Raised if the DataFrame contains columns that are not in the schema, or
        is missing columns that should be in the schema.
    """
    missing_columns = set(_MANIFEST_SCHEMA).difference(set(df.columns))
    extra_columns = set(df.columns) - set(_MANIFEST_SCHEMA)

    if len(missing_columns) > 0:
        raise Exception(
            f"The following columns are missing from the output DataFrame: {', '.join(missing_columns)}"
        )

    if len(extra_columns) > 0:
        raise Exception(
            f"The following columns are in the DataFrame but not in the schema: {', '.join(extra_columns)}"
        )
