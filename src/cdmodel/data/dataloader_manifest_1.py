from typing import Final, NamedTuple, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cdmodel.data.dataset_manifest_1 import ConversationData


class BatchedConversationData(NamedTuple):
    features: Tensor
    embeddings: Tensor
    embeddings_segment_len: Tensor
    num_segments: Tensor
    gender: list[list[str]]
    gender_idx: Tensor
    speaker_id: list[list[int]]
    speaker_id_idx: Tensor
    speaker_id_partner: list[list[int]]
    speaker_id_partner_idx: Tensor
    side_a_speaker_id: list[int]
    side_a_speaker_id_idx: Tensor
    side_b_speaker_id: list[int]
    side_b_speaker_id_idx: Tensor
    da_category: Optional[list[list[str]]] = None
    da_category_idx: Optional[Tensor] = None
    da_consolidated: Optional[list[list[str]]] = None
    da_consolidated_idx: Optional[Tensor] = None


def collate_fn(batches: list[ConversationData]) -> ConversationData:
    features_all: Final[list[Tensor]] = []
    embeddings_all: Final[list[Tensor]] = []
    embeddings_segment_len_all: Final[list[Tensor]] = []
    num_segments_all: Final[list[Tensor]] = []
    gender_all: Final[list[list[str]]] = []
    gender_idx_all: Final[list[Tensor]] = []
    speaker_id_all: Final[list[list[int]]] = []
    speaker_id_idx_all: Final[list[Tensor]] = []
    speaker_id_partner_all: Final[list[list[int]]] = []
    speaker_id_partner_idx_all: Final[list[Tensor]] = []
    side_a_speaker_id_all: Final[list[int]] = []
    side_a_speaker_id_idx_all: Final[list[Tensor]] = []
    side_b_speaker_id_all: Final[list[int]] = []
    side_b_speaker_id_idx_all: Final[list[Tensor]] = []

    # Optional: Dialogue acts
    has_da: bool = False
    da_category_all: Final[list[list[str]]] = []
    da_category_idx_all: Final[list[Tensor]] = []
    da_consolidated_all: Final[list[list[str]]] = []
    da_consolidated_idx_all: Final[list[Tensor]] = []

    # For padding embedding segments
    longest_embedding_segment: int = 0

    for batch in batches:
        features_all.append(batch.features)
        embeddings_all.append(batch.embeddings)
        embeddings_segment_len_all.append(batch.embeddings_segment_len)
        num_segments_all.append(batch.num_segments)
        gender_all.append(batch.gender)
        gender_idx_all.append(batch.gender_idx)
        speaker_id_all.append(batch.speaker_id)
        speaker_id_idx_all.append(batch.speaker_id_idx)
        speaker_id_partner_all.append(batch.speaker_id_partner)
        speaker_id_partner_idx_all.append(batch.speaker_id_partner_idx)
        side_a_speaker_id_all.append(batch.side_a_speaker_id)
        side_a_speaker_id_idx_all.append(batch.side_a_speaker_id_idx)
        side_b_speaker_id_all.append(batch.side_b_speaker_id)
        side_b_speaker_id_idx_all.append(batch.side_b_speaker_id_idx)

        if batch.da_category is not None:
            has_da = True
            da_category_all.append(batch.da_category)
            da_category_idx_all.append(batch.da_category_idx)
            da_consolidated_all.append(batch.da_consolidated)
            da_consolidated_idx_all.append(batch.da_consolidated_idx)

        max_embeddings_len: int = batch.embeddings_segment_len.max().item()
        if longest_embedding_segment < max_embeddings_len:
            longest_embedding_segment = max_embeddings_len

    features: Final[Tensor] = nn.utils.rnn.pad_sequence(features_all, batch_first=True)

    # Embeddings are stored in a 3-dimensional tensor with the following dimensions:
    #
    #    (conversation segments, words, embedding dimension)
    #
    # To make it possible for all turns can be encoded in parallel, all segments from
    # all conversations are concatenated along the first axis. After encoding, it is the
    # model's responsibility to break up the result back into individual conversations.
    embeddings: Final[Tensor] = torch.cat(
        [
            F.pad(x, (0, 0, 0, longest_embedding_segment - x.shape[1]))
            for x in embeddings_all
        ],
        dim=0,
    )

    # Similarly to how the embeddings are represented, the embedding segment lengths are
    # also concatenated along a single dimension. It is the model's responsibility to divide
    # sequences of lengths into individual conversations.
    embeddings_segment_len: Final[Tensor] = torch.cat(embeddings_segment_len_all, dim=0)

    num_segments: Final[Tensor] = torch.cat(num_segments_all, dim=0)

    gender: Final[list[list[str]]] = gender_all
    gender_idx: Final[Tensor] = nn.utils.rnn.pad_sequence(
        gender_idx_all, batch_first=True
    )

    speaker_id: Final[list[list[int]]] = speaker_id_all
    speaker_id_idx: Final[Tensor] = nn.utils.rnn.pad_sequence(
        speaker_id_idx_all, batch_first=True
    )
    speaker_id_partner: Final[list[list[int]]] = speaker_id_partner_all
    speaker_id_partner_idx: Final[Tensor] = nn.utils.rnn.pad_sequence(
        speaker_id_partner_idx_all, batch_first=True
    )

    side_a_speaker_id: Final[list[int]] = side_a_speaker_id_all
    side_a_speaker_id_idx: Final[Tensor] = torch.cat(side_a_speaker_id_idx_all, dim=0)
    side_b_speaker_id: Final[list[int]] = side_b_speaker_id_all
    side_b_speaker_id_idx: Final[Tensor] = torch.cat(side_b_speaker_id_idx_all, dim=0)

    da_category: Optional[list[list[int]]] = None
    da_category_idx: Optional[Tensor] = None
    da_consolidated: Optional[list[list[int]]] = None
    da_consolidated_idx: Optional[Tensor] = None
    if has_da:
        da_category = da_category_all
        da_category_idx = nn.utils.rnn.pad_sequence(
            da_category_idx_all, batch_first=True
        )
        da_consolidated = da_consolidated_all
        da_consolidated_idx = nn.utils.rnn.pad_sequence(
            da_consolidated_idx_all, batch_first=True
        )

    return BatchedConversationData(
        features=features,
        embeddings=embeddings,
        embeddings_segment_len=embeddings_segment_len,
        num_segments=num_segments,
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
