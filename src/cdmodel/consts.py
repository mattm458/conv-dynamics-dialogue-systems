from typing import Final

FEATURES: Final[list[str]] = [
    "duration",
    "duration_vcd",
    "pitch_mean",
    "pitch_5",
    "pitch_95",
    "pitch_range",
    "pitch_mean_log",
    "pitch_5_log",
    "pitch_95_log",
    "pitch_range_log",
    "intensity_mean",
    "intensity_mean_vcd",
    "jitter",
    "shimmer",
    "nhr",
    "nhr_vcd",
    "rate",
    "rate_vcd",
]

NORM_BY_CONV_SPEAKER_POSTFIX: Final[str] = "_norm_by_conv_speaker"
FEATURES_NORM_BY_CONV_SPEAKER: Final[list[str]] = [
    f"{x}_{NORM_BY_CONV_SPEAKER_POSTFIX}" for x in FEATURES
]
