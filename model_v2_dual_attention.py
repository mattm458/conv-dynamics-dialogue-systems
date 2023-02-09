# %%
import csv
from os import path
from typing import List, Tuple

import matplotlib
import numpy as np
import pandas as pd
from lightning import pytorch as pl
import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from data.dataloader import collate_fn
from data.dataset import ConversationDataset
from model.components import Decoder, EmbeddingEncoder, Encoder
from model.util import init_hidden, get_hidden_vector


# %%
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


# %%
class ConversationDataset(Dataset):
    def __init__(
        self,
        conversation_ids,
        embeddings_dir,
        conversation_data_dir="fisher-ipu-data",
        features=FEATURES,
    ):
        super().__init__()
        self.conversation_ids = conversation_ids
        self.embeddings_dir = embeddings_dir
        self.conversation_data_dir = conversation_data_dir
        self.features = features

    def __len__(self):
        return len(self.conversation_ids)

    def __getitem__(self, i):
        conv_id = self.conversation_ids[i]
        conv_data = torch.load(path.join(self.conversation_data_dir, f"{conv_id}.pt"))

        features = conv_data["features"]
        speakers = conv_data["speakers"]

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

        return (
            features,
            speakers,
            embeddings,
            embeddings_len,
            torch.tensor(predict),
            conv_len,
            y,
            torch.LongTensor([len(y)]),
        )


train_ids = open("fisher_test_ids.csv").read().split("\n")
ds = ConversationDataset(train_ids, "fisher-embeddings")
ds[0]


# %%
def collate_fn(batch):
    features_all = []
    speakers_all = []
    embeddings_all = []
    embeddings_len_all = []
    predict_all = []
    conv_len_all = []
    batch_id_all = []
    longest_embeddings = 0

    y_all = []
    y_len_all = []

    for batch_id, (
        features,
        speakers,
        embeddings,
        embeddings_len,
        predict,
        conv_len,
        y,
        y_len,
    ) in enumerate(batch):
        features_all.append(features)
        speakers_all.append(speakers)
        embeddings_all.append(embeddings)
        embeddings_len_all.append(embeddings_len)

        max_embeddings_len = embeddings_len.max().item()
        if longest_embeddings < max_embeddings_len:
            longest_embeddings = max_embeddings_len

        conv_len_all.append(conv_len)
        batch_id_all.extend([batch_id] * conv_len)
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

    return (
        features_all,
        speakers_all,
        embeddings_all,
        embeddings_len_all,
        predict_all,
        conv_len_all,
        batch_id_all,
        y_all,
        y_len_all,
    )


val_dataset = ConversationDataset(
    open("fisher_val_ids.csv").read().split("\n"),
    "fisher-embeddings",
)
dl = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)
for x in dl:
    print(x[5])
    break


# %%
def save_figure_to_numpy(fig):
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    return data


# %%
def save_figure_to_numpy(fig):
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    return data


# %%
class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.ModuleList(
            [nn.GRUCell(in_dim, hidden_dim)]
            + [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )

        self.dropout = nn.Dropout(dropout)

    def get_hidden(self, batch_size, device):
        return [
            torch.zeros((batch_size, self.hidden_dim), device=device)
            for x in range(self.num_layers)
        ]

    def forward(
        self,
        encoder_input: Tensor,
        hidden: List[Tensor],
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        if len(hidden) != len(self.rnn):
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        x = encoder_input

        new_hidden: List[Tuple[Tensor, Tensor]] = []

        for i, rnn in enumerate(self.rnn):
            h = hidden[i]

            h_out = rnn(x, h)
            x = h_out

            if i < (len(self.rnn) - 1):
                x = self.dropout(x)

            new_hidden.append(h_out)

        return x, new_hidden


class Attention(nn.Module):
    def __init__(self, history_in_dim: int, context_dim: int, att_dim: int):
        super().__init__()

        self.context_dim = context_dim

        self.history = nn.Linear(history_in_dim, att_dim, bias=False)
        self.context = nn.Linear(context_dim, att_dim, bias=False)
        self.v = nn.Linear(att_dim, 1, bias=False)

    def forward(
        self, history: Tensor, context: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        history_att = self.history(history)
        context_att = self.context(context).unsqueeze(1)
        score = self.v(torch.tanh(history_att + context_att))

        score = score.masked_fill(mask, float("-inf"))
        score = torch.softmax(score, dim=1)

        score = score.masked_fill(mask, 0.0)
        score_out = score.detach().clone()

        score = score.squeeze(-1).unsqueeze(1)
        att_applied = torch.bmm(score, history)
        att_applied = att_applied.squeeze(1)

        return att_applied, score_out


class Decoder(nn.Module):
    def __init__(
        self,
        decoder_in_dim: int,
        decoder_dropout: float,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.rnn = nn.ModuleList(
            [nn.GRUCell(decoder_in_dim, hidden_dim)]
            + [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )

        self.dropout = nn.Dropout(decoder_dropout)

        linear_arr = [nn.Linear(hidden_dim, output_dim)]
        if activation == "tanh":
            print("Decoder: Tanh activation")
            linear_arr.append(nn.Tanh())

        self.linear = nn.Sequential(*linear_arr)

    def get_hidden(self, batch_size, device):
        return [
            torch.zeros((batch_size, self.hidden_dim), device=device)
            for x in range(self.num_layers)
        ]

    def forward(
        self,
        encoded: Tensor,
        hidden: List[Tuple[Tensor, Tensor]],
        hidden_idx: Tensor,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]], Tensor]:
        if len(hidden) != self.num_layers:
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        batch_size = encoded.shape[0]
        device = encoded.device
        dtype = encoded.dtype

        new_hidden: List[Tuple[Tensor, Tensor]] = []

        x = encoded[hidden_idx]

        for i, rnn in enumerate(self.rnn):
            h = hidden[i]

            h_out = rnn(x, h[hidden_idx])
            x = h_out

            if i < (len(self.rnn) - 1):
                x = self.dropout(x)

            h_new = h.clone().type(h_out.dtype)
            h_new[hidden_idx] = h_out
            new_hidden.append(h_new)

        x = self.linear(x)

        output = torch.zeros(
            (batch_size, self.output_dim), device=device, dtype=x.dtype
        )
        output[hidden_idx] = x

        return output, new_hidden


# %%
import traceback


class ConversationModel(pl.LightningModule):
    def __init__(
        self,
        feature_names=[
            "pitch_mean",
            "pitch_range",
            "intensity",
            "jitter",
            "shimmer",
            "nhr",
            "rate",
        ],
        lr=0.001,
    ):
        super().__init__()

        self.num_features = 7
        self.feature_names = feature_names
        self.lr = lr

        self.embedding_encoder = EmbeddingEncoder(
            embedding_dim=50,
            encoder_out_dim=32,
            encoder_num_layers=2,
            encoder_dropout=0.5,
            attention_dim=32,
        )

        self.encoder = Encoder(
            in_dim=7 + 2 + 32, hidden_dim=32, num_layers=2, dropout=0.5
        )

        self.our_attentions = nn.ModuleList()
        self.their_attentions = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(7):
            self.our_attentions.append(
                Attention(history_in_dim=32, context_dim=32 + (32 * 2), att_dim=32)
            )
            self.their_attentions.append(
                Attention(history_in_dim=32, context_dim=32 + (32 * 2), att_dim=32)
            )
            self.decoders.append(
                Decoder(
                    decoder_in_dim=32 * 2,
                    hidden_dim=32,
                    num_layers=2,
                    decoder_dropout=0.5,
                    output_dim=1,
                    activation=None,
                )
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        (
            features,
            speakers,
            embeddings,
            embeddings_len,
            predict,
            conv_len,
            batch_id,
            y,
            y_len,
        ) = batch

        batch_size = features.shape[0]
        device = features.device

        our_features_pred, our_scores, their_scores = self.sequence(
            features,
            speakers,
            embeddings,
            embeddings_len,
            predict,
            conv_len,
        )
        y_mask = torch.arange(y.shape[1], device=device).unsqueeze(0)
        y_mask = y_mask.repeat(batch_size, 1)
        y_mask = y_mask < y_len.unsqueeze(1)

        loss = F.mse_loss(our_features_pred[predict], y[y_mask])
        loss_l1 = F.smooth_l1_loss(our_features_pred[predict], y[y_mask])

        self.log("validation_loss", loss, on_epoch=True, on_step=False)
        self.log("validation_loss_l1", loss_l1, on_epoch=True, on_step=False)

        return {
            "loss": loss,
            "our_attention_scores": our_scores,
            "their_attention_scores": their_scores,
            "predict": predict,
        }

    def validation_epoch_end(self, outputs):
        our_scores = outputs[0]["our_attention_scores"]
        their_scores = outputs[0]["their_attention_scores"]

        predict = outputs[0]["predict"]

        pred_show = predict[0]

        for scores, label in zip([our_scores, their_scores], ["our", "their"]):
            scores_show = scores[0][pred_show]

            for feature_idx, feature in enumerate(self.feature_names):
                fig, axs = plt.subplots(
                    ncols=1, nrows=scores_show.shape[0], figsize=(10, 10)
                )
                for i, s in enumerate(scores_show.cpu()):
                    ax = axs[i]
                    ax.imshow(
                        s.unsqueeze(0)[:, feature_idx, : i + 1].numpy(),
                        interpolation="nearest",
                        vmin=0 if i == 0 else None,
                        vmax=1 if i == 0 else None,
                        aspect="auto",
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

                fig.canvas.draw()
                data = save_figure_to_numpy(fig)
                plt.close(fig)

                self.logger.experiment.add_image(
                    f"val_alignment_{label}_{feature}",
                    data,
                    self.current_epoch,
                    dataformats="HWC",
                )

    def training_step_end(self, outputs):
        if self.global_step % 500 == 0:
            for name, parameter in self.named_parameters():
                self.logger.experiment.add_histogram(name, parameter, self.global_step)

    def training_step(self, batch, batch_idx):
        (
            features,
            speakers,
            embeddings,
            embeddings_len,
            predict,
            conv_len,
            batch_id,
            y,
            y_len,
        ) = batch

        batch_size = features.shape[0]
        device = features.device

        our_features_pred, our_scores, their_scores = self.sequence(
            features,
            speakers,
            embeddings,
            embeddings_len,
            predict,
            conv_len,
        )
        y_mask = torch.arange(y.shape[1], device=device).unsqueeze(0)
        y_mask = y_mask.repeat(batch_size, 1)
        y_mask = y_mask < y_len.unsqueeze(1)

        loss = F.mse_loss(our_features_pred[predict], y[y_mask])

        self.log("training_loss", loss.detach(), on_epoch=True, on_step=True)

        return loss

    def sequence(
        self,
        features,
        speakers,
        embeddings,
        embeddings_len,
        predict,
        conv_len,
        autoregress_prob=1.0,
    ):
        batch_size = features.shape[0]
        device = features.device
        num_timesteps = features.shape[1]

        embeddings_encoded, _ = self.embedding_encoder(embeddings, embeddings_len)
        embeddings_encoded = nn.utils.rnn.pad_sequence(
            torch.split(embeddings_encoded, conv_len.tolist()), batch_first=True
        )

        features = [x.squeeze(1) for x in torch.split(features, 1, dim=1)]
        speakers = [x.squeeze(1) for x in torch.split(speakers, 1, dim=1)]
        predict = [x.squeeze(1) for x in torch.split(predict, 1, dim=1)]
        embeddings_encoded = [
            x.squeeze(1) for x in torch.split(embeddings_encoded, 1, dim=1)
        ]

        encoder_hidden = self.encoder.get_hidden(batch_size=batch_size, device=device)
        decoder_hidden = [
            decoder.get_hidden(batch_size, device=device) for decoder in self.decoders
        ]

        history = []

        our_history_mask = []
        their_history_mask = []

        decoded_all = []
        our_scores_all = []
        their_scores_all = []
        scores_len_all = []

        prev_features = torch.zeros((batch_size, self.num_features), device=device)
        prev_predict = torch.zeros((batch_size,), device=device, dtype=torch.bool)

        for i in range(num_timesteps - 1):
            autoregress_mask = prev_predict * (
                torch.rand(prev_predict.shape, device=device) < autoregress_prob
            )

            features_timestep = features[i].clone()
            speakers_timestep = speakers[i]
            embeddings_encoded_timestep = embeddings_encoded[i]
            embeddings_encoded_timestep_next = embeddings_encoded[i + 1]

            features_timestep[autoregress_mask] = (
                prev_features[autoregress_mask]
                .detach()
                .clone()
                .type(features_timestep.dtype)
            )

            predict_timestep = predict[i]

            encoder_in = [
                features_timestep,
                speakers_timestep,
                embeddings_encoded_timestep,
            ]
            encoder_in = torch.cat(encoder_in, dim=-1)

            encoded, encoder_hidden = self.encoder(encoder_in, encoder_hidden)
            history.append(encoded.unsqueeze(1))

            our_history_mask.append(
                ~(speakers_timestep == US.to(device))
                .sum(1)
                .type(torch.bool)
                .unsqueeze(1)
            )
            their_history_mask.append(
                ~(speakers_timestep == THEM.to(device))
                .sum(1)
                .type(torch.bool)
                .unsqueeze(1)
            )

            decoded_cat = []
            our_scores_cat = []
            their_scores_cat = []

            for decoder_idx, (our_attention, their_attention, decoder) in enumerate(
                zip(self.our_attentions, self.their_attentions, self.decoders)
            ):
                att_context = torch.cat(decoder_hidden[decoder_idx], dim=-1)
                att_context = torch.cat(
                    [att_context, embeddings_encoded_timestep_next], dim=-1
                )

                history_cat = torch.cat(history, dim=1)

                # OUR HISTORY
                # ================================================
                our_history_att, our_scores = our_attention(
                    history_cat,
                    context=att_context,
                    mask=torch.cat(our_history_mask, dim=1).unsqueeze(2),
                )

                our_scores_expanded = torch.zeros(
                    (batch_size, num_timesteps), device=device
                )
                if predict_timestep.any():
                    our_scores_expanded[
                        predict_timestep, : our_scores.shape[1]
                    ] = our_scores[predict_timestep].squeeze(-1)
                our_scores_cat.append(our_scores_expanded.unsqueeze(1))

                # THEIR HISTORY
                # ================================================
                their_history_att, their_scores = their_attention(
                    history_cat,
                    context=att_context,
                    mask=torch.cat(their_history_mask, dim=1).unsqueeze(2),
                )

                their_scores_expanded = torch.zeros(
                    (batch_size, num_timesteps), device=device
                )
                if predict_timestep.any():
                    their_scores_expanded[
                        predict_timestep, : their_scores.shape[1]
                    ] = their_scores[predict_timestep].squeeze(-1)
                their_scores_cat.append(their_scores_expanded.unsqueeze(1))

                # FINAL DECODING STEP WITH CONCATENATED OUR/THEIR HISTORIES
                # ===================================================
                history_att = torch.cat([our_history_att, their_history_att], dim=-1)

                decoder_in = [history_att, embeddings_encoded_timestep_next]
                decoder_in = torch.cat(decoder_in, dim=-1)
                decoded_out, decoder_hidden_out = decoder(
                    history_att, decoder_hidden[decoder_idx], predict_timestep
                )
                decoder_hidden[decoder_idx] = decoder_hidden_out

                decoded_cat.append(decoded_out)

            our_scores_cat = torch.cat(our_scores_cat, dim=1)
            their_scores_cat = torch.cat(their_scores_cat, dim=1)

            our_scores_all.append(our_scores_cat.unsqueeze(1))
            their_scores_all.append(their_scores_cat.unsqueeze(1))

            decoded_cat = torch.cat(decoded_cat, dim=-1)
            decoded_all.append(decoded_cat.unsqueeze(1))

            prev_predict = predict_timestep
            prev_features = decoded_cat

        decoded_all = torch.cat(decoded_all, dim=1)

        our_scores_all = torch.cat(our_scores_all, dim=1)
        their_scores_all = torch.cat(their_scores_all, dim=1)

        return decoded_all, our_scores_all, their_scores_all


try:
    val_dataset = ConversationDataset(
        open("fisher_val_ids.csv").read().split("\n"),
        "fisher-embeddings",
    )
    dl = DataLoader(val_dataset, batch_size=3, collate_fn=collate_fn)
    model = ConversationModel()
    for batch in dl:
        _ = model.training_step(batch, 0)
        break
except:
    traceback.print_exc()


# %%
train_dataset = ConversationDataset(
    open("fisher_train_ids.csv").read().split("\n"),
    "fisher-embeddings",
)
val_dataset = ConversationDataset(
    open("fisher_val_ids.csv").read().split("\n"),
    "fisher-embeddings",
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    collate_fn=collate_fn,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    num_workers=8,
    persistent_workers=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=64,
    collate_fn=collate_fn,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    num_workers=8,
    persistent_workers=True,
)


# %%
trainer = pl.Trainer(accelerator="gpu", precision=16, devices=[1])
model = ConversationModel()

try:
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,  # ckpt_path='lightning_logs/version_8/checkpoints/epoch=33-step=9928.ckpt'
    )
except:
    traceback.print_exc()



