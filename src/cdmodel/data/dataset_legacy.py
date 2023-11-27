from os import path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import random
from sklearn.preprocessing import OneHotEncoder

FEATURES = [
    "pitch_mean_zscore",
    "pitch_range_zscore",
    "intensity_mean_vcd_zscore",
    "jitter_zscore",
    "shimmer_zscore",
    "nhr_vcd_zscore",
    "rate_zscore",
]

US = torch.tensor([0.0, 1.0])
THEM = torch.tensor([1.0, 0.0])

AGENT = torch.tensor([0.0, 1.0])
HUMAN = torch.tensor([1.0, 0.0])

class LegacyConversationDataset(Dataset):
    def __init__(
        self,
        conversation_ids,
        embeddings_dir,
        conversation_data_dir="fisher-ipu-data",
        features=FEATURES,
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
    ):
        super().__init__()
        self.conversation_ids = conversation_ids
        self.embeddings_dir = embeddings_dir
        self.conversation_data_dir = conversation_data_dir
        self.features = features

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

    def __len__(self):
        return len(self.conversation_ids)

    def __getitem__(self, i):
        conv_id = self.conversation_ids[i]
        conv_data = torch.load(path.join(self.conversation_data_dir, f"{conv_id}.pt"))

        speakers = conv_data["speakers_one_hot"]
        features = []
        for feature in self.features:
            features.append(conv_data[feature])

        features = torch.stack(features, dim=1)

        embeddings = torch.load(
            path.join(self.embeddings_dir, f"{conv_id}-embeddings.pt")
        )
        embeddings_len = torch.load(
            path.join(self.embeddings_dir, f"{conv_id}-lengths.pt")
        )

        speaker_id_idx = torch.tensor(conv_data["speaker_id_idx"]) + 1
        partner_id_idx = torch.tensor(conv_data["partner_id_idx"]) + 1

        if self.zero_pad:
            features = F.pad(features, (0, 0, 1, 0))
            speakers = F.pad(speakers, (0, 0, 1, 0))
            # TODO: Probably not necessary, just include an embeddings_len of 0?
            # embeddings = F.pad(embeddings, (0, 0, 1, 0))
            embeddings_len = F.pad(embeddings_len, (1, 0), value=1)
            embeddings = F.pad(embeddings, (0, 0, 0, 0, 1, 0))

            speaker_id_idx = F.pad(speaker_id_idx, (1, 0))
            partner_id_idx = F.pad(partner_id_idx, (1, 0))

        # Determine which of the speakers is the agent
        agent = US
        human = THEM
        first_pred_padded = False
        if (
            self.agent_assignment == "random" and random.random() >= 0.5
        ) or self.agent_assignment == "first":
            agent = THEM
            human = US
            first_pred_padded = True

        speaker_role = []
        is_agent_turn = []
        for j in range(len(speakers)):
            if torch.equal(speakers[j], agent):
                speaker_role.append(AGENT)
                is_agent_turn.append(True)
            elif torch.equal(speakers[j], human):
                speaker_role.append(HUMAN)
                is_agent_turn.append(False)
            else:
                speaker_role.append(torch.tensor([0.0, 0.0]))
                is_agent_turn.append(False)

        predict = []
        y = []
        for j in range(len(speaker_role) - 1):
            if is_agent_turn[j + 1]:
                predict.append(True)
                y.append(features[j + 1].unsqueeze(0))
            else:
                predict.append(False)

        is_agent_turn = torch.tensor(is_agent_turn)
        speaker_role = torch.stack(speaker_role)
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

        return output
