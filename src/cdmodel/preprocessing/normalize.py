import pandas as pd
from pandas import DataFrame, Series
from cdmodel.preprocessing.consts import NORM_BY_CONV_SPEAKER_POSTFIX


def norm_by_conv_speaker(data: DataFrame, features: list[str]) -> DataFrame:
    def _do_norm(x: Series) -> Series:
        T_MAX = 1
        T_MIN = -1

        median = x.median()
        std = x.std()

        minimum = median - (3 * std)
        maximum = median + (3 * std)

        x = (x - minimum) / (maximum - minimum)
        x = x * (T_MAX - T_MIN) + T_MIN

        return x

    data_norm = data.groupby(["id", "speaker_id"])[features].transform(_do_norm)
    data_norm.columns = pd.Index(
        [f"{x}_{NORM_BY_CONV_SPEAKER_POSTFIX}" for x in features]
    )
    return pd.concat([data, data_norm], axis=1)
