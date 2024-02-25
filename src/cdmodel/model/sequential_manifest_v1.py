from typing import Final, Literal, Optional

import torch
from lightning import pytorch as pl
from torch import Generator, Tensor, nn
from torch.nn import functional as F

from cdmodel.consts import SPEAKER_ROLE_AGENT_IDX, SPEAKER_ROLE_PARTNER_IDX
from cdmodel.data.dataloader_manifest_1 import BatchedConversationData
from cdmodel.model.components import (
    AttentionModule,
    Decoder,
    DualAttention,
    EmbeddingEncoder,
    Encoder,
    NoopAttention,
    SingleAttention,
    SinglePartnerAttention,
)
from cdmodel.model.util import (
    get_role_identity_idx,
    get_speaker_role_idx,
    one_hot_drop_0,
    timestep_split,
)


def expand_cat(x: list[Tensor], dim: int = -1):
    num_dims = None
    max_len = -1
    for t in x:
        if num_dims is None:
            num_dims = len(t.shape)
        elif num_dims != len(t.shape):
            raise Exception("Tensors in x are not of uniform dimensionality!")
        if t.shape[dim] > max_len:
            max_len = t.shape[dim]

    if num_dims is None:
        raise Exception("Cannot expand_cat an empty tensor!")

    if dim == -1:
        dim = num_dims - 1

    padded: list[Tensor] = []
    for t in x:
        padding = [0, 0] * num_dims
        padding[-((dim * 2) + 1)] = max_len - t.shape[dim]
        p = F.pad(t, tuple(padding)).unsqueeze(dim)
        padded.append(p)

    return torch.cat(padded, dim)


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
        attention_style: Literal["dual", "single_both", "single_partner", None],
        lr: float,
        speaker_role_encoding: Literal[None, "one_hot"] = None,
        gender_encoding: Literal[None, "one_hot"] = None,
        gender_dim: Optional[int] = None,
        da_encoding: Literal[None, "one_hot"] = None,
        da_type: Literal[None, "da_category", "da_consolidated"] = None,
        da_dim: Optional[int] = None,
        role_assignment: Literal[
            "agent_first", "agent_second", "random"
        ] = "agent_second",
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
        self.speaker_role_encoding: Final[Literal[None, "one_hot"]] = (
            speaker_role_encoding
        )

        self.gender_encoding: Final[Literal[None, "one_hot"]] = gender_encoding
        self.gender_dim: Final[Optional[int]] = gender_dim
        self.da_encoding: Final[Literal[None, "one_hot", "embedding"]] = da_encoding

        self.validation_predictions: Final[list[Tensor]] = []
        self.validation_attention_ours: Final[list[list[Tensor]]] = []
        self.validation_attention_theirs: Final[list[list[Tensor]]] = []
        self.validation_data: Final[list[BatchedConversationData]] = []

        self.lr: Final[float] = lr
        self.attention_style: Literal["dual", "single_both", "single_partner", None] = (
            attention_style
        )

        self.role_assignment: Final[
            Literal["agent_first", "agent_second", "random"]
        ] = role_assignment
        self.zero_pad: Final[bool] = zero_pad

        self.generator: Optional[Generator] = None

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
            + (decoder_hidden_dim * decoder_num_layers)  # The decoder hidden state
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
        elif attention_style == "single_both":
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
        elif attention_style == "single_partner":
            self.attentions = nn.ModuleList(
                [
                    SinglePartnerAttention(
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
        if num_decoders == 1:
            decoder_out_dim = len(features)
        elif num_decoders == len(features):
            decoder_out_dim = 1
        else:
            raise Exception(
                f"Configuration specifies {num_decoders} which cannot output {len(features)} output features!"
            )

        self.decoders = nn.ModuleList(
            [
                Decoder(
                    decoder_in_dim=decoder_in_dim,
                    hidden_dim=decoder_hidden_dim,
                    num_layers=decoder_num_layers,
                    decoder_dropout=decoder_dropout,
                    output_dim=decoder_out_dim,
                    activation=None,
                )
                for _ in range(num_decoders)
            ]
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def validation_step(self, batch: BatchedConversationData, batch_idx: int):
        if self.role_assignment == "random" and self.generator is not None:
            self.generator.manual_seed(batch.conv_id[0])

        agent_identity_idx, partner_identity_idx = get_role_identity_idx(
            speaker_identity_idx=batch.speaker_id_idx,
            role_assignment=self.role_assignment,
            zero_pad=self.zero_pad,
            generator=self.generator,
        )

        speaker_role_idx = get_speaker_role_idx(
            speaker_identity_idx=batch.speaker_id_idx,
            agent_identity_idx=agent_identity_idx,
            partner_identity_idx=partner_identity_idx,
        )

        predict_next = (speaker_role_idx == SPEAKER_ROLE_AGENT_IDX)[:, 1:]

        agent_features_pred, agent_scores, partner_scores, combined_scores = self(
            features=batch.features,
            speaker_role_idx=speaker_role_idx,
            embeddings=batch.embeddings,
            embeddings_len=batch.embeddings_segment_len,
            predict_next=predict_next,
            conv_len=batch.num_segments,
            gender_idx=batch.gender_idx,
        )

        y = batch.features[:, 1:]

        loss = F.mse_loss(agent_features_pred[predict_next], y[predict_next])
        loss_l1 = F.smooth_l1_loss(agent_features_pred[predict_next], y[predict_next])

        self.log("validation_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("validation_loss_l1", loss_l1, on_epoch=True, on_step=False)

        for feature_idx, feature_name in enumerate(self.features):
            self.log(
                f"validation_loss_l1_{feature_name}",
                F.smooth_l1_loss(
                    agent_features_pred[predict_next][:, feature_idx],
                    y[predict_next][:, feature_idx],
                ),
                on_epoch=True,
                on_step=False,
            )

        self.validation_predictions.append(agent_features_pred)
        self.validation_data.append(batch)
        self.validation_attention_ours.append(agent_scores)
        self.validation_attention_theirs.append(partner_scores)

        return loss

    def on_validation_epoch_end(self):
        self.validation_predictions.clear()
        self.validation_data.clear()
        self.validation_attention_ours.clear()
        self.validation_attention_theirs.clear()

    def on_train_epoch_start(self):
        if self.role_assignment == "random" and self.generator is not None:
            self.generator.seed()

    def on_train_epoch_end(self):
        for name, parameter in self.named_parameters():
            self.logger.experiment.add_histogram(name, parameter, self.global_step)

    def on_fit_start(self):
        if self.role_assignment == "random":
            self.generator = torch.Generator(device=self.device)

    def on_predict_start(self):
        if self.role_assignment == "random":
            self.generator = torch.Generator(device=self.device)

    def training_step(self, batch: BatchedConversationData, batch_idx: int):
        agent_identity_idx: Tensor
        partner_identity_idx: Tensor
        agent_identity_idx, partner_identity_idx = get_role_identity_idx(
            speaker_identity_idx=batch.speaker_id_idx,
            role_assignment=self.role_assignment,
            zero_pad=self.zero_pad,
            generator=self.generator,
        )

        speaker_role_idx: Final[Tensor] = get_speaker_role_idx(
            speaker_identity_idx=batch.speaker_id_idx,
            agent_identity_idx=agent_identity_idx,
            partner_identity_idx=partner_identity_idx,
        )

        predict_next = (speaker_role_idx == SPEAKER_ROLE_AGENT_IDX)[:, 1:]

        agent_features_pred, _, _, _ = self(
            features=batch.features,
            speaker_role_idx=speaker_role_idx,
            embeddings=batch.embeddings,
            embeddings_len=batch.embeddings_segment_len,
            predict_next=predict_next,
            conv_len=batch.num_segments,
            gender_idx=batch.gender_idx,
        )

        y = batch.features[:, 1:]

        loss = F.mse_loss(agent_features_pred[predict_next], y[predict_next])

        self.log(
            "training_loss", loss.detach(), on_epoch=True, on_step=True, prog_bar=True
        )

        for feature_idx, feature_name in enumerate(self.features):
            self.log(
                f"training_loss_l1_{feature_name}",
                F.smooth_l1_loss(
                    agent_features_pred[predict_next][:, feature_idx],
                    y[predict_next][:, feature_idx],
                ).detach(),
                on_epoch=True,
                on_step=True,
            )

        return loss

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def forward(
        self,
        features: Tensor,
        speaker_role_idx: Tensor,
        embeddings: Tensor,
        embeddings_len: Tensor,
        predict_next: Tensor,
        conv_len: Tensor,
        gender_idx: Tensor,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        # Get some basic information from the input tensors
        batch_size: Final[int] = features.shape[0]
        num_segments: Final[int] = features.shape[1]
        device = features.device

        # Bulk encode the texts from all turns, then break them out into appropriate batches
        embeddings_encoded, _ = self.embedding_encoder(embeddings, embeddings_len)
        embeddings_encoded = nn.utils.rnn.pad_sequence(
            torch.split(embeddings_encoded, conv_len.tolist()), batch_first=True  # type: ignore
        )

        speaker_role_encoded: Optional[Tensor] = None
        if self.speaker_role_encoding == "one_hot":
            speaker_role_encoded = one_hot_drop_0(speaker_role_idx, num_classes=3)

        gender_encoded: Optional[Tensor] = None
        if self.gender_encoding == "one_hot":
            gender_encoded = one_hot_drop_0(gender_idx, num_classes=self.gender_dim + 1)

        agent_history_mask = speaker_role_idx == SPEAKER_ROLE_AGENT_IDX
        partner_history_mask = speaker_role_idx == SPEAKER_ROLE_PARTNER_IDX

        # For efficiency, preemptively split some inputs by timestep
        features_segmented = timestep_split(features)
        predict_next_segmented = timestep_split(predict_next)
        speaker_role_encoded_segmented = timestep_split(speaker_role_encoded)
        embeddings_encoded_segmented = timestep_split(embeddings_encoded)

        if self.gender_encoding:
            gender_segmented = timestep_split(gender_encoded)

        # Create initial zero hidden states for the encoder and decoder(s)
        encoder_hidden: list[Tensor] = self.encoder.get_hidden(
            batch_size=batch_size, device=device
        )
        decoder_hidden: list[list[Tensor]] = [
            d.get_hidden(batch_size, device=device) for d in self.decoders
        ]

        # Lists to store accumulated conversation data from the main loop below
        history_cat: list[Tensor] = []
        decoded_all_cat: list[Tensor] = []

        agent_scores_all: list[Tensor] = []
        partner_scores_all: list[Tensor] = []
        combined_scores_all: list[Tensor] = []

        # Placeholders to contain predicted features carried over from the previous timestep
        prev_features = torch.zeros((batch_size, self.num_features), device=device)
        prev_predict = torch.zeros((batch_size,), device=device, dtype=torch.bool)

        # Iterate through the entire conversation
        for i in range(num_segments - 1):
            # Autoregression/teacher forcing:
            # Create a mask that contains True if a previously predicted feature should be fed
            # back into the model, or False if the ground truth value should be used instead
            # (i.e., teacher forcing)
            autoregress_mask = prev_predict & (
                torch.rand(prev_predict.shape, device=device) < 1.0  # autoregress_prob
            )

            features_segment = features_segmented[i].clone()
            features_segment[autoregress_mask] = (
                prev_features[autoregress_mask]
                .detach()
                .clone()
                .type(features_segment.dtype)
            )

            if self.gender_encoding:
                gender_segment = gender_segmented[i]
                gender_segment_next = gender_segmented[i + 1]

            # Assemble the encoder input. This includes the current conversation features
            # and the previously-encoded embeddings.
            encoder_in_arr: list[Tensor] = [
                features_segment,
                embeddings_encoded_segmented[i],
                speaker_role_encoded_segmented[i],
            ]

            if self.gender_encoding:
                encoder_in_arr.append(gender_segment)

            encoder_in: Tensor = torch.cat(encoder_in_arr, dim=-1)

            # Encode the input and append it to the history.
            encoded, encoder_hidden = self.encoder(encoder_in, encoder_hidden)
            history_cat.append(encoded)

            # Concatenate the history tensor and select specific batch indexes where we are predicting
            history = torch.stack(history_cat, dim=1)

            agent_scores_cat: list[Tensor] = []
            partner_scores_cat: list[Tensor] = []
            combined_scores_cat: list[Tensor] = []
            features_pred_cat: list[Tensor] = []

            attention: AttentionModule
            for decoder_idx, (attention, h, decoder) in enumerate(
                zip(
                    self.attentions,
                    decoder_hidden,
                    self.decoders,
                )
            ):
                history_encoded, att_scores = attention(
                    history,
                    context=torch.cat(
                        h + [embeddings_encoded_segmented[i + 1]], dim=-1
                    ),
                    agent_mask=agent_history_mask[:, : i + 1],
                    partner_mask=partner_history_mask[:, : i + 1],
                )

                if att_scores.agent_scores is not None:
                    agent_scores_cat.append(att_scores.agent_scores)
                if att_scores.partner_scores is not None:
                    partner_scores_cat.append(att_scores.partner_scores)
                if att_scores.combined_scores is not None:
                    combined_scores_cat.append(att_scores.combined_scores)

                decoder_in_arr = [
                    history_encoded,
                    embeddings_encoded_segmented[i + 1],
                    speaker_role_encoded_segmented[i + 1],
                ]

                if self.gender_encoding:
                    decoder_in_arr.append(gender_segment_next)

                decoder_in = torch.cat(decoder_in_arr, dim=-1)

                decoder_out, h_out = decoder(decoder_in, h)
                decoder_hidden[decoder_idx] = h_out
                features_pred_cat.append(decoder_out)

            # Assemble final predicted features
            features_pred = torch.cat(features_pred_cat, dim=-1)
            decoded_all_cat.append(features_pred.unsqueeze(1))

            if len(agent_scores_cat) > 0:
                agent_scores_all.append(torch.cat(agent_scores_cat, dim=1))
            if len(partner_scores_cat) > 0:
                partner_scores_all.append(torch.cat(partner_scores_cat, dim=1))
            if len(combined_scores_cat) > 0:
                combined_scores_all.append(torch.cat(combined_scores_cat, dim=1))

            prev_predict = predict_next_segmented[i]
            prev_features = features_pred

        return (
            torch.cat(decoded_all_cat, dim=1),
            (
                expand_cat(agent_scores_all, dim=-1)
                if len(agent_scores_all) > 0
                else None
            ),
            (
                expand_cat(partner_scores_all, dim=-1)
                if len(partner_scores_all) > 0
                else None
            ),
            (
                expand_cat(combined_scores_all, dim=-1)
                if len(combined_scores_all) > 0
                else None
            ),
        )

    def predict_step(self, batch: BatchedConversationData, batch_idx: int) -> tuple[
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        BatchedConversationData,
        Tensor,
        Tensor,
    ]:
        if self.role_assignment == "random" and self.generator is not None:
            self.generator.manual_seed(batch.conv_id[0])

        agent_identity_idx, partner_identity_idx = get_role_identity_idx(
            speaker_identity_idx=batch.speaker_id_idx,
            role_assignment=self.role_assignment,
            zero_pad=self.zero_pad,
            generator=self.generator,
        )

        speaker_role_idx: Final[Tensor] = get_speaker_role_idx(
            speaker_identity_idx=batch.speaker_id_idx,
            agent_identity_idx=agent_identity_idx,
            partner_identity_idx=partner_identity_idx,
        )

        predict_next = (speaker_role_idx == SPEAKER_ROLE_AGENT_IDX)[:, 1:]

        agent_features_pred, agent_scores, partner_scores, combined_scores = self(
            features=batch.features,
            speaker_role_idx=speaker_role_idx,
            embeddings=batch.embeddings,
            embeddings_len=batch.embeddings_segment_len,
            predict_next=predict_next,
            conv_len=batch.num_segments,
            gender_idx=batch.gender_idx,
        )

        return (
            agent_features_pred,
            agent_scores,
            partner_scores,
            combined_scores,
            batch,
            predict_next,
            speaker_role_idx,
        )
