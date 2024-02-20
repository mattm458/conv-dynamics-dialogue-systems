# import pandas as pd
# import pytest

# from cdmodel.preprocessing import (
#     _norm_by_conversation_speaker,
#     _zscore_by_conversation_speaker,
# )


# def test_zscore_by_conversation_speaker():
#     # Sample DataFrame
#     df = pd.DataFrame(
#         [
#             {"id": 1, "speaker_id": 1, "feature": 100},
#             {"id": 1, "speaker_id": 2, "feature": 100},
#             {"id": 1, "speaker_id": 1, "feature": -100},
#             {"id": 1, "speaker_id": 2, "feature": -100},
#             {"id": 2, "speaker_id": 2, "feature": 50},
#             {"id": 2, "speaker_id": 3, "feature": 50},
#             {"id": 2, "speaker_id": 2, "feature": -50},
#             {"id": 2, "speaker_id": 3, "feature": -50},
#         ]
#     )

#     # Perform normalization
#     df_norm = pd.concat(
#         [df, _zscore_by_conversation_speaker(df, features=["feature"])], axis=1
#     )

#     # We should have new z-score normalized feature columns
#     assert "feature_zscore_by_conv_speaker" in df_norm.columns

#     # Per conversation ID and speaker ID, each feature should have mean 0 and unit sd
#     for _, group in df_norm.groupby(["id", "speaker_id"]):
#         assert group["feature_zscore_by_conv_speaker"].mean() == 0
#         assert group["feature_zscore_by_conv_speaker"].std() == pytest.approx(1.0)


# def test_norm_by_conversation_speaker():
#     # Sample DataFrame
#     df = pd.DataFrame(
#         [
#             {"id": 1, "speaker_id": 1, "feature": 100},
#             {"id": 1, "speaker_id": 2, "feature": 100},
#             {"id": 1, "speaker_id": 1, "feature": -100},
#             {"id": 1, "speaker_id": 2, "feature": -100},
#             {"id": 2, "speaker_id": 2, "feature": 50},
#             {"id": 2, "speaker_id": 3, "feature": 50},
#             {"id": 2, "speaker_id": 2, "feature": -50},
#             {"id": 2, "speaker_id": 3, "feature": -50},
#         ]
#     )

#     # Perform normalization
#     df_norm = pd.concat(
#         [df, _norm_by_conversation_speaker(df, features=["feature"])], axis=1
#     )

#     # We should have new normalized feature columns
#     assert "feature_norm_by_conv_speaker" in df_norm.columns
