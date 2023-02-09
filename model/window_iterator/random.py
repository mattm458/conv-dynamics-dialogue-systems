import random
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


class RandomWindowIterator:
    def __init__(
        self,
        window_size: int,
        features: Tensor,
        embeddings: Tensor,
        embeddings_len: Tensor,
        speakers: Tensor,
        predict: Tensor,
    ):
        self.window_size = window_size

        self.features = F.pad(features, (0, 0, window_size - 1, 0))
        self.embeddings = F.pad(embeddings, (0, 0, 0, 0, window_size - 1, 0))
        self.embeddings_len = F.pad(embeddings_len, (window_size - 1, 0))
        self.speakers = F.pad(speakers, (0, 0, window_size - 1, 0))
        self.predict = predict

        self.done = False

    def update(
        self,
        autoregress_input: Optional[Tensor] = None,
    ):
        self.done = True

    def has_next(self) -> bool:
        return not self.done

    def next(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        predict_idx = random.randint(0, self.predict.shape[1] - 1)

        predict_timestep = self.predict[:, predict_idx]

        window_start = predict_idx
        window_end = predict_idx + self.window_size

        features_window = self.features[:, window_start:window_end]
        features_y = self.features[:, window_end]

        speakers_window = self.speakers[:, window_start:window_end]
        speakers_next = self.speakers[:, window_end]

        embeddings_window = self.embeddings[:, window_start:window_end]
        embeddings_window = torch.cat(
            [x.squeeze(1) for x in embeddings_window.split(1, 1)], dim=0
        )
        embeddings_next = self.embeddings[:, window_end]

        embeddings_len_window = self.embeddings_len[:, window_start:window_end]
        embeddings_len_window = torch.cat(
            [x.squeeze(1) for x in embeddings_len_window.split(1, 1)], dim=0
        )
        embeddings_len_next = self.embeddings_len[:, window_end]

        return (
            features_window,
            speakers_window,
            embeddings_window,
            embeddings_len_window,
            features_y,
            speakers_next,
            embeddings_next,
            embeddings_len_next,
            predict_timestep,
        )
