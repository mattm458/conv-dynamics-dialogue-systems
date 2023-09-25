from os import path

import torch
from torch.utils.data import Dataset

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


class ConversationDataset(Dataset):
    def __init__(
        self,
        conversation_ids,
        embeddings_dir,
        conversation_data_dir="fisher-ipu-data",
        features=FEATURES,
        da=False,
    ):
        super().__init__()
        self.conversation_ids = conversation_ids
        self.embeddings_dir = embeddings_dir
        self.conversation_data_dir = conversation_data_dir
        self.features = features

        self.da = da

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

        predict = []
        y = []
        for j in range(len(speakers) - 1):
            if torch.equal(speakers[j + 1], US):
                predict.append(True)
                y.append(features[j + 1].unsqueeze(0))
            else:
                predict.append(False)

        y = torch.cat(y, dim=0)

        conv_len = len(features)

        output = [
            features,
            speakers,
            embeddings,
            embeddings_len,
            torch.tensor(predict),
            conv_len,
            y,
            torch.LongTensor([len(y)]),
        ]

        if self.da:
            output.append(conv_data["da"])

        return tuple(output)
