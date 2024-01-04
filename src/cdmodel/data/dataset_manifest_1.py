import random
from os import path
from typing import Final, Optional

import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.nn import functional as F
from torch.utils.data import Dataset

from cdmodel.data.manifest import get_dataset_version

FEATURES: Final[list[str]] = [
    "pitch_mean_zscore",
    "pitch_range_zscore",
    "intensity_mean_vcd_zscore",
    "jitter_zscore",
    "shimmer_zscore",
    "nhr_vcd_zscore",
    "rate_zscore",
]

AGENT = [0.0, 1.0]
PARTNER = [1.0, 0.0]

_MANIFEST_VERSION = 1


class ConversationDataset(Dataset):
    def __init__(
        self,
        dir: str,
        ids: list[int],
        speaker_ids: dict[int, int],
        features: list[str] = FEATURES,
        gender=False,
        da=False,
        da_transform=None,
        da_encode=False,
        da_tags=None,
        speaker_identity=False,
        speaker_identity_them=True,
        speaker_identity_always_us=False,
        zero_pad=False,
        agent_assignment="second",
        spectrogram_agent=False,
        spectrogram_partner=False,
        spectrogram_dir=None,
        speaker_id_encode_override: Optional[list] = None,
    ):
        if _MANIFEST_VERSION != get_dataset_version(dir):
            raise Exception("Dataset version mismatch!")

        super().__init__()

        self.dir = dir
        self.ids = ids
        self.features = features
        self.speaker_ids = speaker_ids

        if speaker_identity_always_us and speaker_identity_them:
            raise Exception(
                "Parameters 'speaker_identity_always_us' and 'speaker_identity_them' cannot be used together!"
            )

        self.gender = gender
        self.da = da
        self.da_transform = da_transform
        self.speaker_identity = speaker_identity
        self.speaker_identity_them = speaker_identity_them
        self.speaker_identity_always_us = speaker_identity_always_us

        if da_encode and not da_tags:
            raise Exception("Encoding dialogue acts requires a list of valid tags")
        self.da_encode = da_encode
        if da_encode:
            self.da_encoder = OneHotEncoder(sparse_output=False)
            self.da_encoder.fit([[x] for x in da_tags])

        self.zero_pad = zero_pad

        if agent_assignment not in ["first", "random", "second"]:
            raise Exception(
                "agent_assignment must be one of 'first', 'random', or 'second'!"
            )

        self.agent_assignment = agent_assignment

        if (spectrogram_agent or spectrogram_partner) and spectrogram_dir is None:
            raise Exception("If loading a spectrogram, spectrogram_dir is required!")
        self.spectrogram_agent = spectrogram_agent
        self.spectrogram_partner = spectrogram_partner
        self.spectrogram_dir = spectrogram_dir

        self.speaker_id_encode_override = False
        if speaker_id_encode_override is not None:
            self.speaker_id_encode_override = True
            self.speaker_id_override_encoder = LabelEncoder()
            self.speaker_id_override_encoder.fit(speaker_id_encode_override)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        conv_id = self.ids[i]
        conv_data = torch.load(path.join(self.dir, "segments", f"{conv_id}.pt"))

        features = []
        for feature in self.features:
            features.append(conv_data[feature])

        features = torch.tensor(features).swapaxes(0, 1)

        embeddings = torch.load(
            path.join(self.dir, "embeddings", f"{conv_id}-embeddings.pt")
        )
        embeddings_len = torch.load(
            path.join(self.dir, "embeddings", f"{conv_id}-lengths.pt")
        )

        speaker_id_idx = torch.tensor(
            [self.speaker_ids[x] for x in conv_data["speaker_id"]]
        )
        partner_id_idx = torch.tensor(
            [self.speaker_ids[x] for x in conv_data["speaker_id_partner"]]
        )

        # Determine which of the speakers is the agent
        if (
            self.agent_assignment == "random" and random.random() >= 0.5
        ) or self.agent_assignment == "first":
            agent_id = conv_data["speaker_id"][0]
            partner_id = conv_data["speaker_id_partner"][0]
        else:
            agent_id = conv_data["speaker_id_partner"][0]
            partner_id = conv_data["speaker_id"][0]

        speaker_role = torch.tensor(
            [AGENT if x == agent_id else PARTNER for x in conv_data["speaker_id"]]
        )
        is_agent_turn = torch.tensor(
            [True if x == agent_id else False for x in conv_data["speaker_id"]]
        )

        if self.speaker_id_encode_override:
            speaker_id_idx = torch.from_numpy(
                self.speaker_id_override_encoder.transform(conv_data["speaker_id"]) + 1
            )
            partner_id_idx = torch.from_numpy(
                self.speaker_id_override_encoder.transform(
                    conv_data["speaker_id_partner"]
                )
                + 1
            )

        if self.zero_pad:
            features = F.pad(features, (0, 0, 1, 0))
            speaker_role = F.pad(speaker_role, (0, 0, 1, 0))
            # TODO: Probably not necessary, just include an embeddings_len of 0?
            # embeddings = F.pad(embeddings, (0, 0, 1, 0))
            embeddings_len = F.pad(embeddings_len, (1, 0), value=1)
            embeddings = F.pad(embeddings, (0, 0, 0, 0, 1, 0))

            speaker_id_idx = F.pad(speaker_id_idx, (1, 0))
            partner_id_idx = F.pad(partner_id_idx, (1, 0))

            is_agent_turn = F.pad(is_agent_turn, (1, 0))

        predict = []
        y = []
        for j in range(len(speaker_role) - 1):
            if is_agent_turn[j + 1]:
                predict.append(True)
                y.append(features[j + 1].unsqueeze(0))
            else:
                predict.append(False)

        predict = torch.tensor(predict)
        y = torch.cat(y, dim=0)

        conv_len = len(features)

        output = {
            "features": features,
            "speakers": speaker_role,
            "embeddings": embeddings,
            "predict": predict,
            "embeddings_len": embeddings_len,
            "conv_len": conv_len,
            "y": y,
            "y_len": torch.LongTensor([len(y)]),
        }

        if self.da:
            da = conv_data["da"]

            if self.da_transform:
                da = [self.da_transform(x) for x in da]

            if self.da_encode:
                da = self.da_encoder.transform([[x] for x in da])

            if self.zero_pad:
                raise Exception("Not implemented yet!")
                # output["da"] = conv_data["da"]

            output["da"] = torch.tensor(da)

        if self.gender:
            if self.zero_pad:
                raise Exception("Not implemented yet!")
                # output["gender"] = conv_data["gender_one_hot"]
            else:
                output["gender"] = conv_data["gender_one_hot"]

        if self.speaker_identity:
            speaker_identity = speaker_id_idx
            partner_identity = partner_id_idx

            if not self.speaker_identity_them:
                if self.speaker_identity_always_us:
                    speaker_identity[:] = speaker_identity[is_agent_turn][0]
                    partner_identity[:] = speaker_identity[is_agent_turn][0]
                else:
                    speaker_identity[~is_agent_turn] = 0
                    partner_identity[is_agent_turn] = 0

            output["speaker_identity"] = speaker_identity
            output["partner_identity"] = partner_identity

        if self.spectrogram_agent or self.spectrogram_partner:
            spectrogram = torch.load(path.join(self.spectrogram_dir, f"{conv_id}.pt"))
            if self.spectrogram_agent:
                output["spectrogram_agent"] = spectrogram[agent_id]
                output["spectrogram_agent_len"] = torch.LongTensor(
                    [len(spectrogram[agent_id])]
                )
            if self.spectrogram_partner:
                output["spectrogram_partner"] = spectrogram[partner_id]
                output["spectrogram_partner_len"] = torch.LongTensor(
                    [len(spectrogram[partner_id])]
                )

        return output
