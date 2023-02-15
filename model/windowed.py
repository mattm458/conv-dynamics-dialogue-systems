import csv
import traceback
from os import path
from typing import List, Optional, Tuple

import torch
from lightning import pytorch as pl
from torch import nn
from torch.nn import functional as F

from model.components import EmbeddingEncoder, NoopAttention
from model.window_iterator.random import RandomWindowIterator
from model.window_iterator.sequential import SequentialWindowIterator


class WindowedConversationModel(pl.LightningModule):
    def __init__(
        self,
        feature_names,
        embedding_dim,
        embedding_encoder_out_dim,
        embedding_encoder_num_layers,
        embedding_encoder_dropout,
        embedding_encoder_att_dim,
        encoder_hidden_dim,
        encoder_num_layers,
        num_decoders,
        encode_embeddings,
        encode_speaker,
        decoder_context_embeddings,
        decoder_context_speaker,
        lr,
        window_size,
        autoregressive_training,
        training_window_mode,
    ):
        super().__init__()

        self.automatic_optimization = False

        self.save_hyperparameters()

        self.embedding_dim = embedding_dim
        self.lr = lr
        self.window_size = window_size
        print("WindowedConversationModel: Window size of 10")

        self.decoder_context_embeddings = decoder_context_embeddings
        self.decoder_context_speaker = decoder_context_speaker
        self.autoregressive_training = autoregressive_training

        self.training_window_mode = training_window_mode
        print(
            f"WindowedConversationModel: Training window mode is {training_window_mode}"
        )

        self.encode_embeddings = encode_embeddings
        self.encode_speaker = encode_speaker

        encoder_in_dim = len(feature_names)

        if encode_embeddings:
            print("WindowedConversationModel: Encoding embeddings")
            encoder_in_dim += embedding_encoder_out_dim

            self.embedding_encoder = EmbeddingEncoder(
                embedding_dim=embedding_dim,
                encoder_out_dim=embedding_encoder_out_dim,
                encoder_num_layers=embedding_encoder_num_layers,
                encoder_dropout=embedding_encoder_dropout,
                attention_dim=embedding_encoder_att_dim,
                pack_sequence=False,
            )
        else:
            print("WindowedConversationModel: Not encoding embeddings")

        if encode_speaker:
            print("WindowedConversationModel: Encoding speaker")
            encoder_in_dim += 2
        else:
            print("WindowedConversationModel: Not encoding speaker")

        print(f"WindowedCoversationModel: Encoder input dimension is {encoder_in_dim}")

        self.encoder = nn.GRU(
            encoder_in_dim,
            encoder_hidden_dim // 2,
            batch_first=True,
            num_layers=encoder_num_layers,
            bidirectional=True,
        )

        decoder_in_dim = encoder_hidden_dim
        if decoder_context_embeddings:
            print(
                "WindowedConversationModel: Decoder context includes upcoming embeddings"
            )
            decoder_in_dim += embedding_encoder_out_dim
        else:
            print(
                "WindowedConversationModel: Decoder context does not include upcoming embeddings"
            )

        if decoder_context_speaker:
            print(
                "WindowedConversationModel: Decoder context includes upcoming speaker"
            )
            decoder_in_dim += 2
        else:
            print(
                "WindowedConversationModel: Decoder context does not include upcoming speaker"
            )

        print(f"WindowedConversationModel: Decoder input dimension is {decoder_in_dim}")

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(decoder_in_dim, encoder_hidden_dim),
                    nn.ELU(),
                    nn.Linear(encoder_hidden_dim, 1),
                )
                for _ in range(num_decoders)
            ]
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    # def training_step_end(self, outputs):
    #     if self.global_step % 500 == 0:
    #         for name, parameter in self.named_parameters():
    #             self.logger.experiment.add_histogram(name, parameter, self.global_step)

    def forward(self, features, context):
        encoder_in = features
        encoded, _ = self.encoder(encoder_in)

        decoder_out = []
        for decoder in self.decoders:
            decoder_in = [encoded[:, -1]] + context
            decoder_in = torch.cat(decoder_in, dim=-1)

            decoder_out.append(decoder(decoder_in))

        output = torch.cat(decoder_out, dim=1)
        return output

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

        embeddings = nn.utils.rnn.pad_sequence(
            torch.split(embeddings, conv_len.tolist()), batch_first=True
        )
        embeddings_len = nn.utils.rnn.pad_sequence(
            torch.split(embeddings_len, conv_len.tolist()), batch_first=True
        )

        seq = SequentialWindowIterator(
            features=features,
            embeddings=embeddings,
            embeddings_len=embeddings_len,
            speakers=speakers,
            window_size=self.window_size,
            predict=predict,
        )

        while seq.has_next():
            (
                features_window,
                speakers_window,
                embeddings_window,
                embeddings_len_window,
                features_y,
                speakers_next,
                embeddings_next,
                embeddings_len_next,
                predict_timestep,
            ) = seq.next()

            if not predict_timestep.any():
                seq.update()
                continue

            encoder_in = [features_window]

            if self.encode_embeddings:
                embeddings_window_encoded, _ = self.embedding_encoder(
                    embeddings_window, lengths=embeddings_len_window
                )
                embeddings_window_encoded = torch.stack(
                    torch.split(embeddings_window_encoded, batch_size, 0), dim=1
                )

                encoder_in.append(embeddings_window_encoded)

            if self.encode_speaker:
                encoder_in.append(speakers_window)

            encoder_in = torch.concat(encoder_in, dim=-1)

            context = []

            if self.decoder_context_embeddings:
                embeddings_upcoming_encoded, _ = self.embedding_encoder(
                    embeddings_next, lengths=embeddings_len_next
                )
                context.append(embeddings_upcoming_encoded)

            if self.decoder_context_speaker:
                context.append(speakers_next)

            features_pred = self(encoder_in, context=context)

            loss = F.mse_loss(
                features_pred[predict_timestep], features_y[predict_timestep]
            )
            loss_l1 = F.smooth_l1_loss(
                features_pred[predict_timestep], features_y[predict_timestep]
            )

            self.log("validation_loss", loss, on_epoch=True, on_step=False)
            self.log("validation_loss_l1", loss_l1, on_epoch=True, on_step=False)

            if self.current_epoch > 0 and self.current_epoch % 10 == 0:
                # Evaluation with autoregression
                encoder_in = [seq.next_autoregress()]

                if self.encode_embeddings:
                    encoder_in.append(embeddings_window_encoded)
                if self.encode_speaker:
                    encoder_in.append(speakers_window)

                encoder_in = torch.concat(encoder_in, dim=-1)

                context = []
                if self.decoder_context_embeddings:
                    context.append(embeddings_upcoming_encoded)
                if self.decoder_context_speaker:
                    context.append(speakers_next)

                features_pred_autoregress = self(encoder_in, context=context)

                loss = F.mse_loss(
                    features_pred_autoregress[predict_timestep],
                    features_y[predict_timestep],
                )
                loss_l1 = F.smooth_l1_loss(
                    features_pred_autoregress[predict_timestep],
                    features_y[predict_timestep],
                )

                self.log(
                    "validation_loss_autoregress", loss, on_epoch=True, on_step=False
                )
                self.log(
                    "validation_loss_l1_autoregress",
                    loss_l1,
                    on_epoch=True,
                    on_step=False,
                )

            seq.update(autoregress_input=features_pred.detach().clone())

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
        try:
            opt = self.optimizers()
        except:
            opt = self.configure_optimizers()

        batch_size = features.shape[0]

        embeddings = nn.utils.rnn.pad_sequence(
            torch.split(embeddings, conv_len.tolist()), batch_first=True
        )
        embeddings_len = nn.utils.rnn.pad_sequence(
            torch.split(embeddings_len, conv_len.tolist()), batch_first=True
        )

        if self.training_window_mode == "random":
            seq = RandomWindowIterator(  # SequentialWindowIterator(
                features=features,
                embeddings=embeddings,
                embeddings_len=embeddings_len,
                speakers=speakers,
                window_size=self.window_size,
                predict=predict,
            )
        elif self.training_window_mode == "sequential":
            raise Exception("Sequential training window not implemented")
        else:
            raise Exception(
                f"Unknown training window method {self.training_window_mode}"
            )

        while seq.has_next():
            (
                features_window,
                speakers_window,
                embeddings_window,
                embeddings_len_window,
                features_y,
                speakers_next,
                embeddings_next,
                embeddings_len_next,
                predict_timestep,
            ) = seq.next()

            if not predict_timestep.any():
                seq.update()
                continue

            encoder_in = [features_window]

            if self.encode_embeddings:
                embeddings_window_encoded, _ = self.embedding_encoder(
                    embeddings_window, lengths=embeddings_len_window
                )
                embeddings_window_encoded = torch.stack(
                    torch.split(embeddings_window_encoded, batch_size, 0), dim=1
                )

                encoder_in.append(embeddings_window_encoded)

            if self.encode_speaker:
                encoder_in.append(speakers_window)

            encoder_in = torch.concat(encoder_in, dim=-1)

            context = []

            if self.decoder_context_embeddings:
                embeddings_upcoming_encoded, _ = self.embedding_encoder(
                    embeddings_next, lengths=embeddings_len_next
                )
                context.append(embeddings_upcoming_encoded)

            if self.decoder_context_speaker:
                context.append(speakers_next)

            opt.zero_grad(set_to_none=True)
            features_pred = self(encoder_in, context=context)

            loss = F.mse_loss(
                features_pred[predict_timestep], features_y[predict_timestep]
            )
            loss_l1 = F.smooth_l1_loss(
                features_pred[predict_timestep], features_y[predict_timestep]
            )

            self.log("training_loss", loss, on_epoch=True, on_step=True)
            self.log("training_loss_l1", loss_l1, on_epoch=True, on_step=True)

            try:
                self.manual_backward(loss)
                opt.step()
            except Exception as e:
                pass

            seq.update(autoregress_input=features_pred.detach().clone())

    def predict_step(self, batch, batch_idx):
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

        embeddings = nn.utils.rnn.pad_sequence(
            torch.split(embeddings, conv_len.tolist()), batch_first=True
        )
        embeddings_len = nn.utils.rnn.pad_sequence(
            torch.split(embeddings_len, conv_len.tolist()), batch_first=True
        )

        seq = SequentialWindowIterator(
            features=features,
            embeddings=embeddings,
            embeddings_len=embeddings_len,
            speakers=speakers,
            window_size=self.window_size,
            predict=predict,
            keep_all=True,
        )

        while seq.has_next():
            (
                features_window,
                speakers_window,
                embeddings_window,
                embeddings_len_window,
                features_y,
                speakers_next,
                embeddings_next,
                embeddings_len_next,
                predict_timestep,
            ) = seq.next()

            if not predict_timestep.any():
                seq.update()
                continue

            encoder_in = [seq.next_autoregress()]

            if self.encode_embeddings:
                embeddings_window_encoded, _ = self.embedding_encoder(
                    embeddings_window, lengths=embeddings_len_window
                )
                embeddings_window_encoded = torch.stack(
                    torch.split(embeddings_window_encoded, batch_size, 0), dim=1
                )

                encoder_in.append(embeddings_window_encoded)

            if self.encode_speaker:
                encoder_in.append(speakers_window)

            encoder_in = torch.concat(encoder_in, dim=-1)

            context = []

            if self.decoder_context_embeddings:
                embeddings_upcoming_encoded, _ = self.embedding_encoder(
                    embeddings_next, lengths=embeddings_len_next
                )
                context.append(embeddings_upcoming_encoded)

            if self.decoder_context_speaker:
                context.append(speakers_next)

            features_pred = self(encoder_in, context=context)

            seq.update(autoregress_input=features_pred)

        return seq.get_all()
