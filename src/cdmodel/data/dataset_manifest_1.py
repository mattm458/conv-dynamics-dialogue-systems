from os import path
from typing import Final, NamedTuple, Optional, final

import pandas as pd
import torch
import ujson
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from cdmodel.data.manifest import get_dataset_properties, get_dataset_version

FEATURES: Final[list[str]] = [
    "pitch_mean_zscore",
    "pitch_range_zscore",
    "intensity_mean_vcd_zscore",
    "jitter_zscore",
    "shimmer_zscore",
    "nhr_vcd_zscore",
    "rate_zscore",
]

_MANIFEST_VERSION = 1


class ConversationData(NamedTuple):
    conv_id: int
    features: Tensor
    embeddings: Tensor
    embeddings_segment_len: Tensor
    num_segments: Tensor
    gender: list[str]
    gender_idx: Tensor
    speaker_id: list[int]
    speaker_id_idx: Tensor
    speaker_id_partner: list[int]
    speaker_id_partner_idx: Tensor
    side_a_speaker_id: int
    side_a_speaker_id_idx: Tensor
    side_b_speaker_id: int
    side_b_speaker_id_idx: Tensor
    da_category: Optional[list[str]] = None
    da_category_idx: Optional[Tensor] = None
    da_consolidated: Optional[list[str]] = None
    da_consolidated_idx: Optional[Tensor] = None


@final
class ConversationDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        conv_ids: list[int],
        speaker_ids: dict[int, int],
        features: list[str] = FEATURES,
        zero_pad: bool = False,
    ):
        if _MANIFEST_VERSION != get_dataset_version(dataset_dir):
            raise Exception("Dataset version mismatch!")

        super().__init__()

        self.dataset_dir: Final[str] = dataset_dir
        self.conv_id: Final[list[int]] = conv_ids
        self.features: Final[list[str]] = features
        self.speaker_ids: Final[list[int]] = speaker_ids
        self.zero_pad: Final[bool] = zero_pad

        # Additional configurations based on the dataset itself
        properties: Final[dict] = get_dataset_properties(dataset_dir)
        self.has_da: Final[bool] = properties["has_da"]

        # If the dataset contains dialogue acts, load the dialogue act index mappings
        if self.has_da:
            self.da_category_idx: Final[dict[str, int]] = pd.read_csv(
                path.join(dataset_dir, "da_category.csv"), index_col="da"
            )["idx"].to_dict()
            self.da_consolidated_idx: Final[dict[str, int]] = pd.read_csv(
                path.join(dataset_dir, "da_consolidated.csv"), index_col="da"
            )["idx"].to_dict()

    def __len__(self) -> int:
        return len(self.conv_id)

    def __getitem__(self, i: int) -> ConversationData:
        conv_id: Final[int] = self.conv_id[i]

        with open(path.join(self.dataset_dir, "segments", f"{conv_id}.json")) as infile:
            conv_data: Final[dict] = ujson.load(infile)

        features: Tensor = torch.tensor(
            [conv_data[feature] for feature in self.features]
        ).swapaxes(0, 1)

        embeddings: Tensor = torch.load(
            path.join(self.dataset_dir, "embeddings", f"{conv_id}-embeddings.pt")
        )
        embeddings_len: Tensor = torch.load(
            path.join(self.dataset_dir, "embeddings", f"{conv_id}-lengths.pt")
        )

        gender: list[str | None] = conv_data["gender"]
        gender_idx: Tensor = torch.tensor(
            [1 if x == "m" else 2 for x in conv_data["gender"]], dtype=torch.long
        )

        speaker_id: list[int | None] = conv_data["speaker_id"]
        speaker_id_idx: Tensor = torch.tensor(
            [self.speaker_ids[x] for x in conv_data["speaker_id"]],
            dtype=torch.long,
        )

        speaker_id_partner: list[int | None] = conv_data["speaker_id_partner"]
        speaker_id_partner_idx: Tensor = torch.tensor(
            [self.speaker_ids[x] for x in conv_data["speaker_id_partner"]],
            dtype=torch.long,
        )

        side_a_speaker_id: Final[int] = conv_data["side_a_speaker_id"]
        side_a_speaker_id_idx: Final[Tensor] = torch.tensor(
            [self.speaker_ids[side_a_speaker_id]], dtype=torch.long
        )
        side_b_speaker_id: Final[int] = conv_data["side_b_speaker_id"]
        side_b_speaker_id_idx: Final[Tensor] = torch.tensor(
            [self.speaker_ids[side_b_speaker_id]], dtype=torch.long
        )

        da_category: Optional[list[str]] = None
        da_category_idx: Optional[Tensor] = None
        da_consolidated: Optional[list[str]] = None
        da_consolidated_idx: Optional[Tensor] = None

        if self.has_da:
            da_category = conv_data["da_category"]
            da_category_idx = torch.tensor(
                [
                    self.da_category_idx[x] if x is not None else 0
                    for x in conv_data["da_category"]
                ],
                dtype=torch.long,
            )

            da_consolidated = conv_data["da_consolidated"]
            da_consolidated_idx = torch.tensor(
                [
                    self.da_consolidated_idx[x] if x is not None else 0
                    for x in conv_data["da_consolidated"]
                ],
                dtype=torch.long,
            )

        if self.zero_pad:
            features = F.pad(features, (0, 0, 1, 0))

            embeddings = F.pad(embeddings, (0, 0, 0, 0, 1, 0))
            embeddings_len = F.pad(embeddings_len, (1, 0), value=1)

            gender = [None] + gender
            gender_idx = F.pad(gender_idx, (1, 0))

            speaker_id = [None] + speaker_id
            speaker_id_idx = F.pad(speaker_id_idx, (1, 0))

            speaker_id_partner = [None] + speaker_id_partner
            speaker_id_partner_idx = F.pad(speaker_id_partner_idx, (1, 0))

            if self.has_da:
                da_category = [None] + da_category
                da_category_idx = F.pad(da_category_idx, (1, 0))

                da_consolidated = [None] + da_consolidated
                da_consolidated_idx = F.pad(da_consolidated_idx, (1, 0))

        return ConversationData(
            conv_id=conv_id,
            features=features,
            embeddings=embeddings,
            embeddings_segment_len=embeddings_len,
            num_segments=torch.tensor([len(features)], dtype=torch.long),
            gender=gender,
            gender_idx=gender_idx,
            speaker_id=speaker_id,
            speaker_id_idx=speaker_id_idx,
            speaker_id_partner=speaker_id_partner,
            speaker_id_partner_idx=speaker_id_partner_idx,
            side_a_speaker_id=side_a_speaker_id,
            side_a_speaker_id_idx=side_a_speaker_id_idx,
            side_b_speaker_id=side_b_speaker_id,
            side_b_speaker_id_idx=side_b_speaker_id_idx,
            da_category=da_category,
            da_category_idx=da_category_idx,
            da_consolidated=da_consolidated,
            da_consolidated_idx=da_consolidated_idx,
        )
