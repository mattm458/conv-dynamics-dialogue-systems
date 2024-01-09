from typing import Optional

import numpy as np
import torch
from lightning import pytorch as pl
from matplotlib import pyplot as plt
from prosody_modeling.model.prosody_model import ProsodyModel
from torch import Tensor, nn
from torch.nn import functional as F

from cdmodel.model.components import (
    AttentionModule,
    Decoder,
    DualAttention,
    DualCombinedAttention,
    EmbeddingEncoder,
    Encoder,
    NoopAttention,
    SingleAttention,
)
from cdmodel.model.util import timestep_split

US = torch.tensor([0.0, 1.0])
THEM = torch.tensor([1.0, 0.0])


def save_figure_to_numpy(fig):
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    return data


class SequentialConversationModel(pl.LightningModule):
    def __init__(
        self,
        feature_names: list[str],
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
        attention_style: str,
        lr: float,
        gender: bool = False,
        da: bool = False,
        speaker_identity: bool = False,
        speaker_identity_count: Optional[int] = None,  # 520,
        speaker_identity_dim: Optional[int] = None,  # 32,
        speaker_identity_partner: bool = False,
        speaker_identity_encoder: bool = False,
        spectrogram_agent_encoder: bool = False,
        spectrogram_partner_encoder: bool = False,
        spectrogram_agent_encoder_out_dim: Optional[int] = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.validation_outputs: list[Tensor] = []
        self.validation_attention_ours: list[Tensor] = []
        self.validation_attention_ours_history_mask: list[Tensor] = []
        self.validation_attention_theirs: list[Tensor] = []
        self.validation_attention_theirs_history_mask: list[Tensor] = []
        self.validation_data: list[tuple[Tensor, Tensor]] = []

        self.gender = gender
        self.da = da
        self.US = None
        self.THEM = None

        self.num_features = 7
        self.feature_names = feature_names
        self.lr = lr
        self.attention_style = attention_style

        self.speaker_identity = speaker_identity
        self.speaker_identity_partner = speaker_identity_partner
        self.speaker_identity_encoder = speaker_identity_encoder

        self.spectrogram_agent_encoder = spectrogram_agent_encoder
        self.spectrogram_partner_encoder = spectrogram_partner_encoder

        # Set up dimensions
        # =====================
        # Encoder inputs
        encoder_in_dim = len(feature_names) + embedding_encoder_out_dim + 2

        # History: encoder outputs at each timestep
        history_dim = encoder_hidden_dim
        # If we're using speaker identities, they are concatenated with the history
        if speaker_identity and speaker_identity_dim:
            history_dim += speaker_identity_dim

        # Attention context dim: consists of the decoder hidden state,
        # plus encoded embeddings for the next timestep
        att_context_dim = embedding_encoder_att_dim + decoder_hidden_dim

        # Decoder input: Decoder input consits of the following components:
        #   - Attention-summarized historical input, equal to history_dim
        #   - Encoded embeddings for the next timestep
        #   - Speaker role for the next timestep
        att_multiplier = 2 if attention_style == "dual" else 1
        decoder_in_dim = embedding_encoder_att_dim + (history_dim * att_multiplier) + 2

        # Encoders
        # =====================
        # Embedding encoder for textual data
        self.embedding_encoder = EmbeddingEncoder(
            embedding_dim=embedding_dim,
            encoder_out_dim=embedding_encoder_out_dim,
            encoder_num_layers=embedding_encoder_num_layers,
            encoder_dropout=embedding_encoder_dropout,
            attention_dim=embedding_encoder_att_dim,
        )

        # Main encoder for input features
        print(f"Encoder: encoder_hidden_dim = {encoder_hidden_dim}")
        self.encoder = Encoder(
            in_dim=encoder_in_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_num_layers,
            dropout=encoder_dropout,
        )

        # Embeddings
        # =====================
        # If using speaker identities, set up speaker identity embeddings.
        if speaker_identity:
            if speaker_identity_dim is None:
                raise Exception(
                    "If using speaker identity embeddings, speaker_identity_dim is required"
                )
            if speaker_identity_count is None:
                raise Exception(
                    "If using speaker identity embeddings, speaker_identity_count is required"
                )

            self.speaker_identity_embedding = nn.Embedding(
                num_embeddings=speaker_identity_count + 1,
                padding_idx=0,
                embedding_dim=speaker_identity_dim,
            )

        # Attention layers
        # =====================
        print(f"Attention: attention_in_dim = {history_dim}")
        self.attentions = nn.ModuleList()
        for i in range(num_decoders):
            if attention_style == "dual_combined":
                attention: AttentionModule = DualCombinedAttention(
                    history_in_dim=history_dim,
                    context_dim=att_context_dim,
                    att_dim=decoder_att_dim,
                )
            elif attention_style == "dual":
                attention: AttentionModule = DualAttention(
                    history_in_dim=history_dim,
                    context_dim=att_context_dim,
                    att_dim=decoder_att_dim,
                )
            elif attention_style == "single":
                attention: AttentionModule = SingleAttention(
                    history_in_dim=history_dim,
                    context_dim=att_context_dim,
                    att_dim=decoder_att_dim,
                )
            elif attention_style == "noop":
                attention: AttentionModule = NoopAttention()
            else:
                raise Exception(f"Unrecognized attention style '{attention_style}'")

            self.attentions.append(attention)

        # Decoders
        # =====================
        print(f"Decoder: decoder_in_dim = {decoder_in_dim}")
        print(f"Decoder: decoder_hidden_dim =  {decoder_hidden_dim}")
        self.decoders = nn.ModuleList()
        for i in range(num_decoders):
            self.decoders.append(
                Decoder(
                    decoder_in_dim=decoder_in_dim,
                    hidden_dim=decoder_hidden_dim,
                    num_layers=decoder_num_layers,
                    decoder_dropout=decoder_dropout,
                    output_dim=1,
                    activation=None,
                )
            )

        # if spectrogram_agent_encoder or spectrogram_partner_encoder:
        #     if not spectrogram_agent_encoder_out_dim:
        #         raise Exception(
        #             "If using an agent spectrogram encoder, spectrogram_agent_encoder_out_dim is required!"
        #         )

        #     if spectrogram_agent_encoder:
        #         # decoder_attention_context_dim += spectrogram_agent_encoder_out_dim
        #         decoder_in_dim += spectrogram_agent_encoder_out_dim
        #     if spectrogram_partner_encoder:
        #         # decoder_attention_context_dim += spectrogram_agent_encoder_out_dim
        #         decoder_in_dim += spectrogram_agent_encoder_out_dim

        #     self.spectrogram_encoder = ProsodyModel(
        #         conv_out_dim=2560,
        #         rnn_in_dim=768,
        #         use_deltas=True,
        #         rnn_layers=2,
        #         rnn_dropout=0.5,
        #         num_features=spectrogram_agent_encoder_out_dim,
        #     )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        features = batch["features"]
        speakers = batch["speakers"]
        embeddings = batch["embeddings"]
        embeddings_len = batch["embeddings_len"]
        predict = batch["predict"]
        conv_len = batch["conv_len"]

        y = batch["y"]
        y_len = batch["y_len"]

        genders = None
        if self.gender:
            genders = batch["gender"]

        speaker_identities = None
        if self.speaker_identity:
            speaker_identities = batch["speaker_identity"]

        partner_identities = None
        if self.speaker_identity_partner:
            partner_identities = batch["partner_identity"]

        spectrogram_agent = None
        spectrogram_agent_len = None
        if self.spectrogram_agent_encoder:
            spectrogram_agent = batch["spectrogram_agent"]
            spectrogram_agent_len = batch["spectrogram_agent_len"]

        spectrogram_partner = None
        spectrogram_partner_len = None
        if self.spectrogram_partner_encoder:
            spectrogram_partner = batch["spectrogram_partner"]
            spectrogram_partner_len = batch["spectrogram_partner_len"]

        batch_size = features.shape[0]
        device = features.device

        (
            our_features_pred,
            our_scores_all,
            our_history_mask,
            their_scores_all,
            their_history_mask,
        ) = self(
            features,
            speakers,
            embeddings,
            embeddings_len,
            predict,
            conv_len,
            genders=genders,
            speaker_identities=speaker_identities,
            partner_identities=partner_identities,
            agent_spectrogram=spectrogram_agent,
            agent_spectrogram_len=spectrogram_agent_len,
            partner_spectrogram=spectrogram_partner,
            partner_spectrogram_len=spectrogram_partner_len,
        )

        y_mask = torch.arange(y.shape[1], device=device).unsqueeze(0)
        y_mask = y_mask.repeat(batch_size, 1)
        y_mask = y_mask < y_len.unsqueeze(1)

        loss = F.mse_loss(our_features_pred[predict], y[y_mask])
        loss_l1 = F.smooth_l1_loss(our_features_pred[predict], y[y_mask])

        self.log("validation_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("validation_loss_l1", loss_l1, on_epoch=True, on_step=False)

        for feature_idx, feature_name in enumerate(self.feature_names):
            self.log(
                f"validation_loss_l1_{feature_name}",
                F.smooth_l1_loss(
                    our_features_pred[predict][:, feature_idx],
                    y[y_mask][:, feature_idx],
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

        if self.speaker_identity:
            self.logger.experiment.add_embedding(
                self.speaker_identity_embedding.weight,
                tag="Speaker identity",
                global_step=self.global_step,
            )

    def training_step(self, batch, batch_idx):
        features = batch["features"]
        speakers = batch["speakers"]
        embeddings = batch["embeddings"]
        embeddings_len = batch["embeddings_len"]
        predict = batch["predict"]
        conv_len = batch["conv_len"]

        y = batch["y"]
        y_len = batch["y_len"]

        genders = None
        if self.gender:
            genders = batch["gender"]

        speaker_identities = None
        if self.speaker_identity:
            speaker_identities = batch["speaker_identity"]

        partner_identities = None
        if self.speaker_identity_partner:
            partner_identities = batch["partner_identity"]

        spectrogram_agent = None
        spectrogram_agent_len = None
        if self.spectrogram_agent_encoder:
            spectrogram_agent = batch["spectrogram_agent"]
            spectrogram_agent_len = batch["spectrogram_agent_len"]

        spectrogram_partner = None
        spectrogram_partner_len = None
        if self.spectrogram_partner_encoder:
            spectrogram_partner = batch["spectrogram_partner"]
            spectrogram_partner_len = batch["spectrogram_partner_len"]

        batch_size = features.shape[0]
        device = features.device

        our_features_pred, _, _, _, _ = self(
            features,
            speakers,
            embeddings,
            embeddings_len,
            predict,
            conv_len,
            genders=genders,
            speaker_identities=speaker_identities,
            partner_identities=partner_identities,
            agent_spectrogram=spectrogram_agent,
            agent_spectrogram_len=spectrogram_agent_len,
            partner_spectrogram=spectrogram_partner,
            partner_spectrogram_len=spectrogram_partner_len,
            autoregress_prob=0.0,
        )

        y_mask = torch.arange(y.shape[1], device=device).unsqueeze(0)
        y_mask = y_mask.repeat(batch_size, 1)
        y_mask = y_mask < y_len.unsqueeze(1)

        loss = F.mse_loss(our_features_pred[predict], y[y_mask])

        self.log(
            "training_loss", loss.detach(), on_epoch=True, on_step=True, prog_bar=True
        )
        for feature_idx, feature_name in enumerate(self.feature_names):
            self.log(
                f"training_loss_l1_{feature_name}",
                F.smooth_l1_loss(
                    our_features_pred[predict][:, feature_idx],
                    y[y_mask][:, feature_idx],
                ),
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

        if self.speaker_identity:
            speaker_identities = self.speaker_identity_embedding(speaker_identities)
            speaker_identities = timestep_split(speaker_identities)

            if self.speaker_identity_partner:
                partner_identities = self.speaker_identity_embedding(partner_identities)
                partner_identities = timestep_split(partner_identities)

        # Create masks that represent our and their timesteps in the history
        if self.US is None:
            self.US = US.to(device)
        if self.THEM is None:
            self.THEM = THEM.to(device)

        our_history_mask = (speakers == self.US).all(dim=-1)
        their_history_mask = (speakers == self.THEM).all(dim=-1)

        # For efficiency, preemptively split some inputs by timestep
        features = timestep_split(features)
        predict = timestep_split(predict)
        speakers = timestep_split(speakers)
        embeddings_encoded = timestep_split(embeddings_encoded)

        if self.gender:
            genders = timestep_split(genders)

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

        spectrogram_agent_encoded = None
        if self.spectrogram_agent_encoder:
            spectrogram_agent_encoded, _, _, _ = self.spectrogram_encoder(
                agent_spectrogram, agent_spectrogram_len
            )
            spectrogram_agent_encoded = F.tanh(spectrogram_agent_encoded)

        spectrogram_partner_encoded = None
        if self.spectrogram_partner_encoder:
            spectrogram_partner_encoded, _, _, _ = self.spectrogram_encoder(
                partner_spectrogram, partner_spectrogram_len
            )
            spectrogram_partner_encoded = F.tanh(spectrogram_partner_encoded)

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

            speaker_timestep = speakers[i]

            if self.gender:
                gender_timestep = genders[i]
                gender_next = genders[i + 1]

            if self.speaker_identity:
                speaker_identity_timestep = speaker_identities[0]
                speaker_identity_next = speaker_identities[1]

                if self.speaker_identity_partner:
                    partner_identity_timestep = partner_identities[0]
                    partner_identity_next = partner_identities[1]

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

            if self.gender:
                encoder_in.append(gender_timestep)

            if self.speaker_identity and self.speaker_identity_encoder:
                encoder_in.append(speaker_identity_timestep)

            encoder_in = torch.cat(encoder_in, dim=-1)

            # Encode the input and append it to the history.
            encoded, encoder_hidden = self.encoder(encoder_in, encoder_hidden)

            encoded_list = [encoded]
            if self.speaker_identity:
                encoded_list.append(speaker_identity_timestep)

            encoded = torch.concat(encoded_list, dim=1)

            history.append(encoded.unsqueeze(1))

            # Concatenate the history tensor and select specific batch indexes where we are predicting
            history_cat = torch.cat(history, dim=1)

            # Get the encoded representation of the upcoming units we are about to predict
            embeddings_encoded_timestep_next_pred = embeddings_encoded_timestep_next

            # Get the speaker that the decoder is about to receive
            speaker_next = speakers[i + 1]

            # Assemble the attention context vector for each of the decoder attention layer(s)
            att_contexts = []
            for h in decoder_hidden:
                att_contexts_arr = [h[-1], embeddings_encoded_timestep_next_pred]

                # if self.gender:
                #     att_contexts_arr.append(gender_next)
                # if self.speaker_identity:
                #     att_contexts_arr.append(speaker_identity_next)

                #     if self.speaker_identity_partner:
                #         att_contexts_arr.append(partner_identity_next)

                # if self.spectrogram_agent_encoder:
                #     att_contexts_arr.append(spectrogram_agent_encoded)

                # if self.spectrogram_partner_encoder:
                #     att_contexts_arr.append(spectrogram_partner_encoded)

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

                if self.gender:
                    decoder_in_arr.append(gender_next)

                # if self.speaker_identity:
                #     decoder_in_arr.append(speaker_identity_next)
                #     if self.speaker_identity_partner:
                #         decoder_in_arr.append(partner_identity_next)

                # if self.spectrogram_agent_encoder:
                #     decoder_in_arr.append(spectrogram_agent_encoded)

                # if self.spectrogram_partner_encoder:
                #     decoder_in_arr.append(spectrogram_partner_encoded)

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
