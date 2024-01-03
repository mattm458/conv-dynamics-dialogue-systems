import pandas as pd
from pandas import DataFrame, Series
from cdmodel.consts import NORM_BY_CONV_SPEAKER_POSTFIX


def __do_norm(x: Series) -> Series:
    T_MAX = 1
    T_MIN = -1

    median = x.median()
    std = x.std()

    minimum = median - (3 * std)
    maximum = median + (3 * std)

    x = (x - minimum) / (maximum - minimum)
    x = x * (T_MAX - T_MIN) + T_MIN

    return x


def norm_by_conv_speaker(df: DataFrame, features: list[str]) -> DataFrame:
    """
    Normalizes a set of features in a DataFrame by conversation ID and speaker, per [Raitio et al., 2020](https://arxiv.org/abs/2009.06775).

    This method scales values 3 standard deviations below and above the median to [-1, 1].

    Parameters
    ----------
    data : DataFrame
        A DataFrame containing features to be normalized.
    features : list[str]
        A list of features to normalize.

    Returns
    -------
    DataFrame
        A DataFrame containing normalized values.
    """

    data_norm = df.groupby(["id", "speaker_id"])[features].transform(__do_norm)
    data_norm.columns = pd.Index(
        [f"{x}_{NORM_BY_CONV_SPEAKER_POSTFIX}" for x in features]
    )

    return data_norm
