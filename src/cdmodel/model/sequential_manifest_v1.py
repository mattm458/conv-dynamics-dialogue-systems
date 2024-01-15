from typing import Final, Literal, Optional

import numpy as np
import torch
from lightning import pytorch as pl
from torch import Generator, Tensor, nn
from torch.nn import functional as F

from cdmodel.data.dataloader_manifest_1 import BatchedConversationData
from cdmodel.model.components import (
    Decoder,
    DualAttention,
    EmbeddingEncoder,
    Encoder,
    NoopAttention,
    SingleAttention,
)
from cdmodel.model.util import timestep_split

US: Final[int] = 2
THEM: Final[int] = 1


class SequentialConversationModel(pl.LightningModule):
    def __init__(
        self,
        features: list[str],
        embedding_dim: int,
        embedding_encoder_out_dim: int,
        embedding_encoder_num_layers: int,
        embedding_encoder_dropout: float,
        embedding_encoder_att_dim: int,
        encoder_hidden_dim: int,
        encoder_num_layers: int,
        encoder_dropout: float,
        decoder_att_dim: int,
        decoder_hidden_dim: int,
        decoder_num_layers: int,
        decoder_dropout: float,
        num_decoders: int,
        attention_style: Literal["dual_combined", "dual", "single", None],
        lr: float,
        speaker_role_encoding: Literal[None, "one_hot"] = None,
        gender_encoding: Literal[None, "one_hot"] = None,
        gender_dim: Optional[int] = None,
        da_encoding: Literal[None, "one_hot"] = None,
        da_type: Literal[None, "da_category", "da_consolidated"] = None,
        da_dim: Optional[int] = None,
        speaker_agent_role: Literal["first", "second", "random"] = "second",
        zero_pad: bool = False,
    ):
        if gender_encoding is not None and gender_dim is None:
            raise Exception(
                "If the model uses a representation of gender, you must specify gender_dim!"
            )

        if da_encoding is not None and (da_type is None or da_dim is None):
            raise Exception(
                "If the model uses a representation of dialogue acts, you must specify da_type and da_dim!"
            )

        super().__init__()

        self.save_hyperparameters()

        self.features: Final[list[str]] = features
        self.num_features: Final[int] = len(features)
        self.speaker_role_encoding: Final[
            Literal[None, "one_hot"]
        ] = speaker_role_encoding

        self.gender_encoding: Final[Literal[None, "one_hot"]] = gender_encoding
        self.gender_dim: Final[Optional[int]] = gender_dim
        self.da_encoding: Final[Literal[None, "one_hot", "embedding"]] = da_encoding

        self.validation_outputs: Final[list[Tensor]] = []
        self.validation_attention_ours: Final[list[Tensor]] = []
        self.validation_attention_ours_history_mask: Final[list[Tensor]] = []
        self.validation_attention_theirs: Final[list[Tensor]] = []
        self.validation_attention_theirs_history_mask: Final[list[Tensor]] = []
        self.validation_data: Final[list[tuple[Tensor, Tensor]]] = []

        self.lr: Final[float] = lr
        self.attention_style: Final[
            Literal["dual_combined", "dual", "single", None]
        ] = attention_style

        self.speaker_agent_role: Final[
            Literal["first", "second", "random"]
        ] = speaker_agent_role
        self.zero_pad: Final[bool] = zero_pad

        # Embedding Encoder
        # =====================
        # The embedding encoder encodes textual data associated with each conversational
        # segment. At each segment, it accepts a sequence of word embeddings and outputs
        # a vector of size `embedding_encoder_out_dim`.
        self.embedding_encoder = EmbeddingEncoder(
            embedding_dim=embedding_dim,
            encoder_out_dim=embedding_encoder_out_dim,
            encoder_num_layers=embedding_encoder_num_layers,
            encoder_dropout=embedding_encoder_dropout,
            attention_dim=embedding_encoder_att_dim,
        )

        # Segment encoder
        # =====================
        # The segment encoder outputs a representation of each conversational segment. The
        # encoder input includes speech features extracted from the segment, an encoded
        # representation of the words spoken in the segment, and a one-hot vector of
        # the speaker role. Each encoded segment is kept by appending it to the conversation
        # history.
        encoder_in_dim: Final[int] = (
            self.num_features  # The number of speech features the model is predicting
            + embedding_encoder_out_dim  # Dimensions of the embedding encoder output
            + 2  # One-hot speaker role vector
        )

        # Main encoder for input features
        print(f"Encoder: encoder_hidden_dim = {encoder_hidden_dim}")
        self.encoder = Encoder(
            in_dim=encoder_in_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_num_layers,
            dropout=encoder_dropout,
        )

        # The dimensions of each historical timestep as output by the segment encoder.
        history_dim: Final[int] = encoder_hidden_dim

        # Attention
        # =====================
        # The attention mechanisms attend to segments in the conversation history. They
        # determine which segments are most useful for decoding into the upcoming speech
        # features. Depending on the configuration, there may be one or more attention
        # mechanisms and one or more decoders.

        # Each attention mechanism outputs a tensor of the same size as a historical
        # timestep. Calculate the total output size depending on whether
        # we're using dual or single attention
        att_multiplier: Final[int] = 2 if attention_style == "dual" else 1
        att_history_out_dim: Final[int] = history_dim * att_multiplier

        att_context_dim: Final[int] = (
            embedding_encoder_att_dim  # The encoded representation of the upcoming segment transcript
            + decoder_hidden_dim  # The decoder hidden state
        )

        # Initialize the attention mechanisms
        if attention_style == "dual":
            self.attentions = nn.ModuleList(
                [
                    DualAttention(
                        history_in_dim=history_dim,
                        context_dim=att_context_dim,
                        att_dim=decoder_att_dim,
                    )
                    for _ in range(num_decoders)
                ]
            )
        elif attention_style == "single":
            self.attentions = nn.ModuleList(
                [
                    SingleAttention(
                        history_in_dim=history_dim,
                        context_dim=att_context_dim,
                        att_dim=decoder_att_dim,
                    )
                    for _ in range(num_decoders)
                ]
            )
        elif attention_style is None:
            self.attentions = nn.ModuleList(
                [NoopAttention() for _ in range(num_decoders)]
            )
        else:
            raise Exception(f"Unrecognized attention style '{attention_style}'")

        # Decoders
        # =====================
        # Decoders predict speech features from the upcoming segment based on its transcript
        # and from attention-summarized historical segments.

        decoder_in_dim = (
            embedding_encoder_out_dim  # Size of the embedding encoder output for upcoming segment text
            + att_history_out_dim  # Size of the attention mechanism output for summarized historical segments
            + 2  # One-hot speaker role vector
        )

        # Initialize the decoders
        self.decoders = nn.ModuleList(
            [
                Decoder(
                    decoder_in_dim=decoder_in_dim,
                    hidden_dim=decoder_hidden_dim,
                    num_layers=decoder_num_layers,
                    decoder_dropout=decoder_dropout,
                    output_dim=1,
                    activation=None,
                )
                for _ in range(num_decoders)
            ]
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def validation_step(self, batch: BatchedConversationData, batch_idx: int):
        # Establish the speaker role
        if self.speaker_agent_role == "first":
            agent_segment_idx: int = int(self.zero_pad)
            agent_idx: Tensor = batch.speaker_id_idx[:, agent_segment_idx]
            partner_idx: Tensor = batch.speaker_id_idx[:, agent_segment_idx + 1]
        elif self.speaker_agent_role == "second":
            agent_segment_idx: int = int(self.zero_pad) + 1
            agent_idx: Tensor = batch.speaker_id_idx[:, agent_segment_idx]
            partner_idx: Tensor = batch.speaker_id_idx[:, agent_segment_idx - 1]
        elif self.speaker_agent_role == "random":
            generator: Final[Generator] = torch.random.manual_seed(batch.conv_id[0])
            agent_idx_bool: Final[Tensor] = (
                torch.rand(
                    batch.speaker_id_idx.shape[0],
                    generator=generator,
                )
                < 0.5
            ).to(self.device)
            partner_idx_bool: Final[Tensor] = ~agent_idx_bool
            agent_segment_idxs: Tensor = agent_idx_bool.type(torch.long)
            partner_segment_idxs: Tensor = partner_idx_bool.type(torch.long)
            if self.zero_pad:
                agent_segment_idxs += 1
                partner_segment_idxs += 1
            agent_idx = torch.gather(
                batch.speaker_id_idx, 1, agent_segment_idxs.unsqueeze(1)
            ).squeeze(1)
            partner_idx = torch.gather(
                batch.speaker_id_idx, 1, partner_segment_idxs.unsqueeze(1)
            ).squeeze(1)

        is_agent: Final[Tensor] = batch.speaker_id_idx.eq(agent_idx.unsqueeze(1))
        is_partner: Final[Tensor] = batch.speaker_id_idx.eq(partner_idx.unsqueeze(1))
        speaker_role: Final[Tensor] = torch.zeros_like(batch.speaker_id_idx)
        speaker_role[is_agent] = US
        speaker_role[is_partner] = THEM
        predict = (speaker_role == US)[:, 1:]

        (
            our_features_pred,
            our_scores_all,
            our_history_mask,
            their_scores_all,
            their_history_mask,
        ) = self(
            features=batch.features,
            speakers=speaker_role,
            embeddings=batch.embeddings,
            embeddings_len=batch.embeddings_segment_len,
            predict=predict,
            conv_len=batch.num_segments,
            genders=batch.gender_idx,
        )

        y = batch.features[:, 1:]
        y_len = batch.num_segments - 1
        batch_size = batch.features.shape[0]

        loss = F.mse_loss(our_features_pred[predict], y[predict])
        loss_l1 = F.smooth_l1_loss(our_features_pred[predict], y[predict])

        self.log("validation_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("validation_loss_l1", loss_l1, on_epoch=True, on_step=False)

        for feature_idx, feature_name in enumerate(self.features):
            self.log(
                f"validation_loss_l1_{feature_name}",
                F.smooth_l1_loss(
                    our_features_pred[predict][:, feature_idx],
                    y[predict][:, feature_idx],
                ),
                on_epoch=True,
                on_step=False,
            )

        self.validation_outputs.append(our_features_pred)
        self.validation_data.append(batch)
        self.validation_attention_ours.append(our_scores_all)
        self.validation_attention_ours_history_mask.append(our_history_mask)
        self.validation_attention_theirs.append(their_scores_all)
        self.validation_attention_theirs_history_mask.append(their_history_mask)

        return loss

    def on_validation_epoch_end(self):
        # self.validation_outputs.append(our_features_pred)
        # self.validation_data.append((features, speakers, predict))

        torch.save(
            (
                self.validation_outputs,
                self.validation_data,
                self.validation_attention_ours,
                self.validation_attention_ours_history_mask,
                self.validation_attention_theirs,
                self.validation_attention_theirs_history_mask,
            ),
            "validation_data.pt",
        )

        self.validation_outputs.clear()
        self.validation_data.clear()
        self.validation_attention_ours.clear()
        self.validation_attention_ours_history_mask.clear()
        self.validation_attention_theirs.clear()
        self.validation_attention_theirs_history_mask.clear()

    def on_train_epoch_end(self):
        for name, parameter in self.named_parameters():
            self.logger.experiment.add_histogram(name, parameter, self.global_step)

        # if self.speaker_identity:
        #     self.logger.experiment.add_embedding(
        #         self.speaker_identity_embedding.weight,
        #         tag="Speaker identity",
        #         global_step=self.global_step,
        #     )

    def training_step(self, batch, batch_idx):
        # Establish the speaker role
        if self.speaker_agent_role == "first":
            agent_segment_idx: int = int(self.zero_pad)
            agent_idx: Tensor = batch.speaker_id_idx[:, agent_segment_idx]
            partner_idx: Tensor = batch.speaker_id_idx[:, agent_segment_idx + 1]
        elif self.speaker_agent_role == "second":
            agent_segment_idx: int = int(self.zero_pad) + 1
            agent_idx: Tensor = batch.speaker_id_idx[:, agent_segment_idx]
            partner_idx: Tensor = batch.speaker_id_idx[:, agent_segment_idx - 1]
        elif self.speaker_agent_role == "random":
            agent_idx_bool: Final[Tensor] = (
                torch.rand(batch.speaker_id_idx.shape[0]) < 0.5
            ).to(self.device)
            partner_idx_bool: Final[Tensor] = ~agent_idx_bool
            agent_segment_idxs: Tensor = agent_idx_bool.type(torch.long)
            partner_segment_idxs: Tensor = partner_idx_bool.type(torch.long)
            if self.zero_pad:
                agent_segment_idxs += 1
                partner_segment_idxs += 1
            agent_idx = torch.gather(
                batch.speaker_id_idx, 1, agent_segment_idxs.unsqueeze(1)
            ).squeeze(1)
            partner_idx = torch.gather(
                batch.speaker_id_idx, 1, partner_segment_idxs.unsqueeze(1)
            ).squeeze(1)

        is_agent: Final[Tensor] = batch.speaker_id_idx.eq(agent_idx.unsqueeze(1))
        is_partner: Final[Tensor] = batch.speaker_id_idx.eq(partner_idx.unsqueeze(1))
        speaker_role: Final[Tensor] = torch.zeros_like(batch.speaker_id_idx)
        speaker_role[is_agent] = US
        speaker_role[is_partner] = THEM
        predict = (speaker_role == US)[:, 1:]

        (
            our_features_pred,
            our_scores_all,
            our_history_mask,
            their_scores_all,
            their_history_mask,
        ) = self(
            features=batch.features,
            speakers=speaker_role,
            embeddings=batch.embeddings,
            embeddings_len=batch.embeddings_segment_len,
            predict=predict,
            conv_len=batch.num_segments,
            genders=batch.gender_idx,
        )

        y = batch.features[:, 1:]
        y_len = batch.num_segments - 1
        batch_size = batch.features.shape[0]

        loss = F.mse_loss(our_features_pred[predict], y[predict])

        self.log(
            "training_loss", loss.detach(), on_epoch=True, on_step=True, prog_bar=True
        )

        for feature_idx, feature_name in enumerate(self.features):
            self.log(
                f"training_loss_l1_{feature_name}",
                F.smooth_l1_loss(
                    our_features_pred[predict][:, feature_idx],
                    y[predict][:, feature_idx],
                ).detach(),
                on_epoch=True,
                on_step=True,
            )

        return loss

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def forward(
        self,
        features,
        speakers,
        embeddings,
        embeddings_len,
        predict,
        conv_len,
        autoregress_prob=1.0,
        genders=None,
        speaker_identities=None,
        partner_identities=None,
        agent_spectrogram=None,
        agent_spectrogram_len=None,
        partner_spectrogram=None,
        partner_spectrogram_len=None,
    ):
        # Get some basic information from the input tensors
        batch_size = features.shape[0]
        num_timesteps = features.shape[1]
        device = features.device

        # Bulk encode the texts from all turns, then break them out into appropriate batches
        embeddings_encoded, _ = self.embedding_encoder(embeddings, embeddings_len)
        embeddings_encoded = nn.utils.rnn.pad_sequence(
            torch.split(embeddings_encoded, conv_len.tolist()), batch_first=True
        )

        # Encode gender
        if self.gender_encoding == "one_hot":
            gender_encoded: Final[Tensor] = F.one_hot(
                genders, num_classes=self.gender_dim + 1
            )[:, :, 1:]

        if self.speaker_role_encoding == "one_hot":
            speaker_role_encoded: Final[Tensor] = F.one_hot(speakers, num_classes=3)[
                :, :, 1:
            ]

        # if self.speaker_identity:
        #     speaker_identities = self.speaker_identity_embedding(speaker_identities)
        #     speaker_identities = timestep_split(speaker_identities)

        #     if self.speaker_identity_partner:
        #         partner_identities = self.speaker_identity_embedding(partner_identities)
        #         partner_identities = timestep_split(partner_identities)

        our_history_mask = (speakers.unsqueeze(2) == US).all(dim=-1)
        their_history_mask = (speakers.unsqueeze(2) == THEM).all(dim=-1)

        # For efficiency, preemptively split some inputs by timestep
        features = timestep_split(features)
        predict = timestep_split(predict)
        speakers_timesteps = timestep_split(speaker_role_encoded)
        embeddings_encoded = timestep_split(embeddings_encoded)

        if self.gender_encoding:
            gender_timesteps = timestep_split(gender_encoded)

        # Create initial zero hidden states for the encoder and decoder(s)
        encoder_hidden = self.encoder.get_hidden(batch_size=batch_size, device=device)
        decoder_hidden = [
            d.get_hidden(batch_size, device=device) for d in self.decoders
        ]

        # Lists to store accumulated conversation data from the main loop below
        history = []
        decoded_all = []
        our_scores_all = []
        their_scores_all = []

        # Placeholders to contain predicted features carried over from the previous timestep
        prev_features = torch.zeros((batch_size, self.num_features), device=device)
        prev_predict = torch.zeros((batch_size,), device=device, dtype=torch.bool)

        # Iterate through the entire conversation
        for i in range(num_timesteps - 1):
            # Autoregression/teacher forcing:
            # Create a mask that contains True if a previously predicted feature should be fed
            # back into the model, or False if the ground truth value should be used instead
            # (i.e., teacher forcing)
            autoregress_mask = prev_predict * (
                torch.rand(prev_predict.shape, device=device) < 1.0  # autoregress_prob
            )
            features_timestep = features[i].clone()
            features_timestep[autoregress_mask] = (
                prev_features[autoregress_mask]
                .detach()
                .clone()
                .type(features_timestep.dtype)
            )

            speaker_timestep = speakers_timesteps[i]

            if self.gender_encoding:
                gender_timestep = gender_timesteps[i]
                gender_next = gender_timesteps[i + 1]

            # if self.speaker_identity:
            #     speaker_identity_timestep = speaker_identities[0]
            #     speaker_identity_next = speaker_identities[1]

            #     if self.speaker_identity_partner:
            #         partner_identity_timestep = partner_identities[0]
            #         partner_identity_next = partner_identities[1]

            # Get some timestep-specific data from the input
            embeddings_encoded_timestep = embeddings_encoded[i]
            embeddings_encoded_timestep_next = embeddings_encoded[i + 1]
            predict_timestep = predict[i]

            # Assemble the encoder input. This includes the current conversation features
            # and the previously-encoded embeddings.
            encoder_in = [
                features_timestep,
                embeddings_encoded_timestep,
                speaker_timestep,
            ]

            if self.gender_encoding:
                encoder_in.append(gender_timestep)

            # if self.speaker_identity and self.speaker_identity_encoder:
            #     encoder_in.append(speaker_identity_timestep)

            encoder_in = torch.cat(encoder_in, dim=-1)

            # Encode the input and append it to the history.
            encoded, encoder_hidden = self.encoder(encoder_in, encoder_hidden)

            encoded_list = [encoded]
            # if self.speaker_identity:
            #     encoded_list.append(speaker_identity_timestep)

            encoded = torch.concat(encoded_list, dim=1)

            history.append(encoded.unsqueeze(1))

            # Concatenate the history tensor and select specific batch indexes where we are predicting
            history_cat = torch.cat(history, dim=1)

            # Get the encoded representation of the upcoming units we are about to predict
            embeddings_encoded_timestep_next_pred = embeddings_encoded_timestep_next

            # Get the speaker that the decoder is about to receive
            speaker_next = speakers_timesteps[i + 1]

            # Assemble the attention context vector for each of the decoder attention layer(s)
            att_contexts = []
            for h in decoder_hidden:
                att_contexts_arr = [h[-1], embeddings_encoded_timestep_next_pred]
                att_contexts.append(torch.cat(att_contexts_arr, dim=-1))

            # Get our/their history masks for the timesteps we're about to predict
            our_history_mask_timestep = our_history_mask[:, : i + 1]
            their_history_mask_timestep = their_history_mask[:, : i + 1]

            our_scores_cat = []
            their_scores_cat = []
            features_pred = []
            combined_scores_cat = []

            for decoder_idx, (attention, att_context, h, decoder) in enumerate(
                zip(
                    self.attentions,
                    att_contexts,
                    decoder_hidden,
                    self.decoders,
                )
            ):
                history_att, (our_scores, their_scores, combined_scores) = attention(
                    history_cat,
                    context=att_context,
                    our_mask=our_history_mask_timestep,
                    their_mask=their_history_mask_timestep,
                )
                combined_scores_cat.append(combined_scores)

                if our_scores is not None:
                    our_scores_cat.append(our_scores)
                if their_scores is not None:
                    their_scores_cat.append(their_scores)
                if combined_scores is not None:
                    combined_scores_cat.append(combined_scores)

                decoder_in_arr = [
                    history_att,
                    embeddings_encoded_timestep_next_pred,
                    speaker_next,
                ]

                if self.gender_encoding:
                    decoder_in_arr.append(gender_next)

                # if self.speaker_identity:
                #     decoder_in_arr.append(speaker_identity_next)
                #     if self.speaker_identity_partner:
                #         decoder_in_arr.append(partner_identity_next)

                decoder_in = torch.cat(decoder_in_arr, dim=-1)

                decoder_out, h_out = decoder(decoder_in, h)
                decoder_hidden[decoder_idx] = h_out
                features_pred.append(decoder_out)

            # Assemble final predicted features
            features_pred = torch.cat(features_pred, dim=-1)
            decoded_all.append(features_pred.unsqueeze(1))

            prev_predict = predict_timestep
            prev_features = features_pred

            our_scores_expanded_all = []
            their_scores_expanded_all = []

            if len(our_scores_cat) > 0:
                for our_scores in our_scores_cat:
                    our_scores_expanded = torch.zeros(
                        (batch_size, num_timesteps - 1), device=device
                    )
                    our_scores_expanded[:, : our_scores.shape[1]] = our_scores.squeeze(
                        -1
                    )
                    our_scores_expanded_all.append(our_scores_expanded.unsqueeze(1))

                our_scores_expanded_all = torch.cat(our_scores_expanded_all, dim=1)
                our_scores_all.append(our_scores_expanded_all.unsqueeze(1))

            if len(their_scores_cat) > 0:
                for their_scores in their_scores_cat:
                    their_scores_expanded = torch.zeros(
                        (batch_size, num_timesteps - 1), device=device
                    )
                    their_scores_expanded[
                        :, : their_scores.shape[1]
                    ] = their_scores.squeeze(-1)

                    their_scores_expanded_all.append(their_scores_expanded.unsqueeze(1))

                their_scores_expanded_all = torch.cat(their_scores_expanded_all, dim=1)
                their_scores_all.append(their_scores_expanded_all.unsqueeze(1))

        decoded_all = torch.cat(decoded_all, dim=1)

        if len(our_scores_all) > 0:
            our_scores_all = (
                torch.cat(our_scores_all, dim=1) if len(our_scores_all) > 0 else None
            )
        if len(their_scores_all) > 0:
            their_scores_all = (
                torch.cat(their_scores_all, dim=1)
                if len(their_scores_all) > 0
                else None
            )

        return (
            decoded_all,
            our_scores_all,
            our_history_mask,
            their_scores_all,
            their_history_mask,
        )

    def predict_step(self, batch, batch_idx):
        features = batch["features"]
        speakers = batch["speakser"]
        embeddings = batch["embeddings"]
        embeddings_len = batch["embeddings_len"]
        predict = batch["predict"]
        conv_len = batch["conv_len"]

        (
            our_features_pred,
            our_scores,
            our_scores_mask,
            their_scores,
            their_scores_mask,
        ) = self(
            features,
            speakers,
            embeddings,
            embeddings_len,
            predict,
            conv_len,
        )

        output = {
            "y_hat": our_features_pred,
            "y": features,
            "predict": predict,
            "our_attention_scores": our_scores,
            "our_attention_scores_mask": our_scores_mask,
            "their_attention_scores": their_scores,
            "their_attention_scores_mask": their_scores_mask,
            "predict": predict,
            "speakers": speakers,
            "conv_len": conv_len,
        }

        if "da" in batch:
            output["da"] = batch["da"]

        return output
