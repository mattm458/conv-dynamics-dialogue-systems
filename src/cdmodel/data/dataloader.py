import torch
from torch import nn
from torch.nn import functional as F
from functools import partial


def get_collate_fn(
    has_da=False,
    has_gender=False,
    has_speaker_identity=False,
    has_spectrogram_agent=False,
    has_spectrogram_partner=False,
):
    return partial(
        collate_fn,
        has_da=has_da,
        has_gender=has_gender,
        has_speaker_identity=has_speaker_identity,
        has_spectrogram_agent=has_spectrogram_agent,
        has_spectrogram_partner=has_spectrogram_partner,
    )


def collate_fn(
    batches,
    has_da,
    has_gender,
    has_speaker_identity,
    has_spectrogram_agent,
    has_spectrogram_partner,
):
    features_all = []
    speakers_all = []
    embeddings_all = []
    embeddings_len_all = []
    predict_all = []
    conv_len_all = []
    longest_embeddings = 0
    gender_all = []
    da_all = []
    speaker_identity_all = []
    partner_identity_all = []
    spectrogram_agent_all = []
    spectrogram_agent_len_all = []
    spectrogram_partner_all = []
    spectrogram_partner_len_all = []

    y_all = []
    y_len_all = []

    for batch in batches:
        features_all.append(batch["features"])
        speakers_all.append(batch["speakers"])
        embeddings_all.append(batch["embeddings"])
        embeddings_len_all.append(batch["embeddings_len"])

        max_embeddings_len = batch["embeddings_len"].max().item()
        if longest_embeddings < max_embeddings_len:
            longest_embeddings = max_embeddings_len

        conv_len_all.append(batch["conv_len"])
        predict_all.append(batch["predict"])

        y_all.append(batch["y"])
        y_len_all.append(batch["y_len"])

        if has_gender:
            gender_all.append(batch["gender"])

        if has_da:
            da_all.append(batch["da"])

        if has_speaker_identity:
            speaker_identity_all.append(batch["speaker_identity"])
            partner_identity_all.append(batch["partner_identity"])

        if has_spectrogram_agent:
            spectrogram_agent_all.append(batch["spectrogram_agent"])
            spectrogram_agent_len_all.append(batch["spectrogram_agent_len"])

        if has_spectrogram_partner:
            spectrogram_partner_all.append(batch["spectrogram_partner"])
            spectrogram_partner_len_all.append(batch["spectrogram_partner_len"])

    features_all = nn.utils.rnn.pad_sequence(features_all, batch_first=True)
    speakers_all = nn.utils.rnn.pad_sequence(speakers_all, batch_first=True)
    embeddings_all = torch.cat(
        [F.pad(x, (0, 0, 0, longest_embeddings - x.shape[1])) for x in embeddings_all],
        dim=0,
    )
    embeddings_len_all = torch.cat(embeddings_len_all, dim=0)
    predict_all = nn.utils.rnn.pad_sequence(predict_all, batch_first=True)
    conv_len_all = torch.LongTensor(conv_len_all)

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

    if has_da:
        if len(features_all) != len(da_all):
            raise Exception(
                f"There are fewer dialogue acts ({len(da_all)}) than features ({len(features_all)})! There is probably something wrong with the Dataset."
            )

        output["da"] = nn.utils.rnn.pad_sequence(da_all, batch_first=True)

    if has_gender:
        if len(features_all) != len(gender_all):
            raise Exception(
                f"There are fewer gender labels ({len(gender_all)}) than features ({len(features_all)})! There is probably something wrong with the Dataset."
            )

        output["gender"] = nn.utils.rnn.pad_sequence(gender_all, batch_first=True)

    if has_speaker_identity:
        output["speaker_identity"] = nn.utils.rnn.pad_sequence(
            speaker_identity_all, batch_first=True
        )
        output["partner_identity"] = nn.utils.rnn.pad_sequence(
            partner_identity_all, batch_first=True
        )

    if has_spectrogram_agent:
        output["spectrogram_agent"] = nn.utils.rnn.pad_sequence(
            spectrogram_agent_all, batch_first=True
        )
        output["spectrogram_agent_len"] = torch.LongTensor(spectrogram_agent_len_all)

    if has_spectrogram_partner:
        output["spectrogram_partner"] = nn.utils.rnn.pad_sequence(
            spectrogram_partner_all, batch_first=True
        )
        output["spectrogram_partner_len"] = torch.LongTensor(spectrogram_partner_len_all)

    return output
