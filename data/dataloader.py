import torch
from torch import nn
from torch.nn import functional as F


def collate_fn(batches):
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

    has_da = False

    for batch in batches:
        if len(batch) == 9:
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
            ) = batch
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
            ) = batch

        features_all.append(batch["features"])
        speakers_all.append(batch["speakers"])
        embeddings_all.append(batch["embeddings"])
        embeddings_len_all.append(batch["embeddings_len"])

        max_embeddings_len = embeddings_len.max().item()
        if longest_embeddings < max_embeddings_len:
            longest_embeddings = max_embeddings_len

        conv_len_all.append(batch["conv_len"])
        predict_all.append(batch["predict"])

        y_all.append(batch["y"])
        y_len_all.append(batch["y_len"])

        if "da" in batch:
            da_all.append(batch["da"])

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

    output = {
        "features": features_all,
        "speakers": speakers_all,
        "embeddings": embeddings_all,
        "predict": predict_all,
        "embeddings_len": embeddings_len_all,
        "conv_len": conv_len_all,
        "y": y_all,
        "y_len": y_len_all,
    }

    if len(features_all) != len(da_all):
        raise Exception(
            f"There are fewer dialogue acts ({len(da_all)} than features ({len(features_all)})! There is probably something wrong with the Dataset."
        )

    if len(da_all) > 0:
        output["da"] = da_all

    return output
