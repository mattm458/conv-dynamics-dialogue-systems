from typing import Optional, Tuple

import torch
from torch import Tensor


class SequentialWindowIterator:
    def __init__(
        self,
        window_size: int,
        features: Tensor,
        embeddings: Tensor,
        embeddings_len: Tensor,
        speakers: Tensor,
        predict: Tensor,
        keep_all: Tensor = False,
    ):
        batch_size = features.shape[0]
        num_features = features.shape[-1]
        embedding_dim = embeddings.shape[-1]

        device = features.device
        longest_embedding_seq = embeddings_len.max()

        self.i = 1
        self.max_length = features.shape[1]

        self.updated: bool = True
        self.updated_autoregress: bool = True

        # Windowing for turn/IPU feature values
        self.features_split = torch.split(features, 1, 1)
        self.features_window = [
            torch.zeros((batch_size, 1, num_features), device=device)
            for _ in range(window_size - 1)
        ] + [self.features_split[0]]

        # Window for turn/IPU feature values accumulated over the conversation
        # through autoregression
        self.features_window_autoregress = [
            torch.zeros((batch_size, 1, num_features), device=device)
            for _ in range(window_size - 1)
        ] + [self.features_split[0]]

        # Window for turn/IPU-level word embeddings
        self.embeddings_split = [x.squeeze(1) for x in torch.split(embeddings, 1, 1)]
        self.embeddings_window = [
            torch.zeros(
                (batch_size, longest_embedding_seq, embedding_dim), device=device
            )
            for _ in range(window_size - 1)
        ] + [self.embeddings_split[0]]

        # Window for word embedding sequence lengths
        self.embeddings_len_split = [
            x.squeeze(1) for x in torch.split(embeddings_len, 1, 1)
        ]
        self.embeddings_len_window = [
            torch.zeros(batch_size, device=device, dtype=torch.long)
            for _ in range(window_size - 1)
        ] + [self.embeddings_len_split[0]]

        # Window for speaker identity flag
        self.speakers_split = [x for x in torch.split(speakers, 1, 1)]
        self.speakers_window = [
            torch.zeros((batch_size, 1, 2), device=device)
            for _ in range(window_size - 1)
        ] + [self.speakers_split[0]]

        self.predict_split = [x.squeeze(1) for x in torch.split(predict, 1, 1)]

        self.keep_all = keep_all
        if keep_all:
            self.window_all = []

    def update(
        self,
        autoregress_input: Optional[Tensor] = None,
    ):
        self.updated = True
        self.updated_autoregress = True

        i = self.i

        self.features_window.append(self.features_split[i])
        self.features_window.pop(0)

        if autoregress_input is not None:
            x = self.features_split[i].clone()
            predict = self.predict_split[i - 1]
            x[predict] = autoregress_input[predict].unsqueeze(1).type(x.dtype)

            self.features_window_autoregress.append(x)
            if self.keep_all:
                self.window_all.append(x)
        else:
            self.features_window_autoregress.append(self.features_split[i])
            if self.keep_all:
                self.window_all.append(self.features_split[i])

        self.features_window_autoregress.pop(0)

        self.embeddings_window.append(self.embeddings_split[i])
        self.embeddings_window.pop(0)

        self.embeddings_len_window.append(self.embeddings_len_split[i])
        self.embeddings_len_window.pop(0)

        self.speakers_window.append(self.speakers_split[i])
        self.speakers_window.pop(0)

        self.i += 1

    def has_next(self) -> bool:
        return not (self.i == self.max_length)

    def next(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not self.updated:
            raise Exception("Sequential object has not been updated!")

        self.updated = False
        i = self.i

        features_window = torch.cat(self.features_window, dim=1)
        features_y = self.features_split[i].squeeze(1)

        embeddings_window = torch.cat(self.embeddings_window, dim=0)
        embeddings_next = self.embeddings_split[i]

        embeddings_len_window = torch.cat(self.embeddings_len_window, dim=0)
        embeddings_len_next = self.embeddings_len_split[i]

        speakers_window = torch.cat(self.speakers_window, dim=1)
        speakers_next = self.speakers_split[i].squeeze(1)

        predict_timestep = self.predict_split[i - 1]

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

    def next_autoregress(self) -> Tensor:
        return torch.cat(self.features_window_autoregress, dim=1)

    def get_all(self) -> Optional[Tensor]:
        if not self.keep_all:
            return None

        return torch.cat(self.window_all, dim=1)
