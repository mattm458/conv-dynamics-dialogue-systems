import torch
from torch import nn
from torch.nn import functional as F


def collate_fn(batch):
    features_all = []
    speakers_all = []
    embeddings_all = []
    embeddings_len_all = []
    predict_all = []
    conv_len_all = []
    longest_embeddings = 0
    da_all = []

    y_all = []
    y_len_all = []

    for batch_i in batch:
        if len(batch_i) == 9:
            (
                features,
                speakers,
                embeddings,
                embeddings_len,
                predict,
                conv_len,
                y,
                y_len,
                da,
            ) = batch_i
            da_all.append(da)
        else:
            (
                features,
                speakers,
                embeddings,
                embeddings_len,
                predict,
                conv_len,
                y,
                y_len,
            ) = batch_i

        features_all.append(features)
        speakers_all.append(speakers)
        embeddings_all.append(embeddings)
        embeddings_len_all.append(embeddings_len)

        max_embeddings_len = embeddings_len.max().item()
        if longest_embeddings < max_embeddings_len:
            longest_embeddings = max_embeddings_len

        conv_len_all.append(conv_len)
        predict_all.append(predict)

        y_all.append(y)
        y_len_all.append(y_len)

    features_all = nn.utils.rnn.pad_sequence(features_all, batch_first=True)
    speakers_all = nn.utils.rnn.pad_sequence(speakers_all, batch_first=True)
    embeddings_all = torch.cat(
        [F.pad(x, (0, 0, 0, longest_embeddings - x.shape[1])) for x in embeddings_all],
        dim=0,
    )
    embeddings_len_all = torch.cat(embeddings_len_all, dim=0)
    predict_all = nn.utils.rnn.pad_sequence(predict_all, batch_first=True)
    conv_len_all = torch.LongTensor(conv_len_all)
    batch_id_all = torch.LongTensor(batch_id_all)

    y_all = nn.utils.rnn.pad_sequence(y_all, batch_first=True)
    y_len_all = torch.LongTensor(y_len_all)

    output = [
        features_all,
        speakers_all,
        embeddings_all,
        embeddings_len_all,
        predict_all,
        conv_len_all,
        y_all,
        y_len_all,
    ]

    if len(da_all) > 0:
        output.append(da_all)

    return tuple(output)
