from typing import List, Literal, Optional, Tuple, Final

import torch
from torch import Tensor, jit, nn

from cdmodel.model.util import lengths_to_mask


class Encoder(jit.ScriptModule):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()

        self.hidden_dim: Final[int] = hidden_dim
        self.num_layers: Final[int] = num_layers

        self.rnn: Final[nn.ModuleList] = nn.ModuleList(
            [nn.GRUCell(in_dim, hidden_dim)]
            + [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )

        self.dropout: Final[nn.Dropout] = nn.Dropout(dropout)

    def get_hidden(self, batch_size: int, device) -> list[Tensor]:
        return [
            torch.zeros((batch_size, self.hidden_dim), device=device)
            for x in range(self.num_layers)
        ]

    @jit.script_method
    def forward(
        self,
        encoder_input: Tensor,
        hidden: List[Tensor],
    ) -> Tuple[Tensor, List[Tensor]]:
        if len(hidden) != len(self.rnn):
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        x: Tensor = encoder_input
        new_hidden: List[Tensor] = []

        i: int
        rnn: nn.GRUCell
        for i, rnn in enumerate(self.rnn):
            h_out = rnn(x, hidden[i])
            x = h_out
            new_hidden.append(h_out)

            if i < (len(self.rnn) - 1):
                x = self.dropout(x)

        return x, new_hidden


class AttentionModule(jit.ScriptModule):
    pass


class Attention(jit.ScriptModule):
    def __init__(self, history_in_dim: int, context_dim: int, att_dim: int):
        super().__init__()

        self.context_dim = context_dim

        self.history = nn.Linear(history_in_dim, att_dim, bias=False)
        self.context = nn.Linear(context_dim, att_dim, bias=False)
        self.v = nn.Linear(att_dim, 1, bias=False)

    @jit.script_method
    def forward(
        self, history: Tensor, context: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        history_att: Final[Tensor] = self.history(history)
        context_att: Final[Tensor] = self.context(context).unsqueeze(1)

        score: Tensor = self.v(torch.tanh(history_att + context_att))
        score = score.masked_fill(mask, float("-inf"))
        score = torch.softmax(score, dim=1)
        score = score.masked_fill(mask, 0.0)

        att_applied: Final[Tensor] = torch.bmm(
            score.squeeze(-1).unsqueeze(1), history
        ).squeeze(1)

        return att_applied, score.detach().clone()


class DualAttention(AttentionModule):
    def __init__(self, history_in_dim: int, context_dim: int, att_dim: int):
        super().__init__()

        self.our_attention: Final[Attention] = Attention(
            history_in_dim=history_in_dim,
            context_dim=context_dim,
            att_dim=att_dim,
        )
        self.their_attention: Final[Attention] = Attention(
            history_in_dim=history_in_dim,
            context_dim=context_dim,
            att_dim=att_dim,
        )

    @jit.script_method
    def forward(
        self, history: Tensor, context: Tensor, agent_mask: Tensor, partner_mask: Tensor
    ) -> Tuple[Tensor, Tuple[Tensor | None, Tensor | None, Tensor | None]]:
        agent_att, agent_scores = self.our_attention(
            history,
            context=context,
            mask=~agent_mask.unsqueeze(-1),
        )

        partner_att, partner_scores = self.their_attention(
            history,
            context=context,
            mask=~partner_mask.unsqueeze(-1),
        )

        return torch.cat([agent_att, partner_att], dim=-1), (
            agent_scores,
            partner_scores,
            None,
        )


class SingleAttention(AttentionModule):
    def __init__(self, history_in_dim: int, context_dim: int, att_dim: int):
        super().__init__()

        self.attention = Attention(
            history_in_dim=history_in_dim,
            context_dim=context_dim,
            att_dim=att_dim,
        )

    @jit.script_method
    def forward(
        self, history: Tensor, context: Tensor, agent_mask: Tensor, partner_mask: Tensor
    ) -> Tuple[Tensor, Tuple[Tensor | None, Tensor | None, Tensor | None]]:
        att, scores = self.attention(
            history,
            context=context,
            mask=~(agent_mask + partner_mask).unsqueeze(-1),
        )

        return att, (scores, None, None)


class SinglePartnerAttention(AttentionModule):
    def __init__(self, history_in_dim: int, context_dim: int, att_dim: int):
        super().__init__()

        self.attention = Attention(
            history_in_dim=history_in_dim,
            context_dim=context_dim,
            att_dim=att_dim,
        )

    @jit.script_method
    def forward(
        self, history: Tensor, context: Tensor, agent_mask: Tensor, partner_mask: Tensor
    ) -> Tuple[Tensor, Tuple[Tensor | None, Tensor | None, Tensor | None]]:
        att, scores = self.attention(
            history,
            context=context,
            mask=~(partner_mask).unsqueeze(-1),
        )

        return att, (scores, None, None)


class NoopAttention(AttentionModule):
    @jit.script_method
    def forward(
        self, history: Tensor, context: Tensor, agent_mask: Tensor, partner_mask: Tensor
    ) -> Tuple[Tensor, Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]]:
        return history[:, -1], (None, None, None)


class EmbeddingEncoder(jit.ScriptModule):
    def __init__(
        self,
        embedding_dim: int,
        encoder_out_dim: int,
        encoder_num_layers: int,
        encoder_dropout: float,
        attention_dim: int,
        pack_sequence=True,
    ):
        super().__init__()

        lstm_out_dim = encoder_out_dim // 2

        self.encoder_out_dim = encoder_out_dim
        self.encoder_num_layers = encoder_num_layers
        self.pack_sequence = pack_sequence

        self.encoder = nn.GRU(
            embedding_dim,
            lstm_out_dim,
            bidirectional=True,
            num_layers=encoder_num_layers,
            dropout=encoder_dropout,
            batch_first=True,
        )
        self.encoder.flatten_parameters()

        self.attention = Attention(
            history_in_dim=encoder_out_dim,
            context_dim=encoder_out_dim * 2,
            att_dim=attention_dim,
        )

    @jit.script_method
    def forward(self, encoder_in: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = encoder_in.shape[0]

        if self.pack_sequence:
            encoder_in_packed = nn.utils.rnn.pack_padded_sequence(
                encoder_in,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            encoder_out_tmp, h = self.encoder(encoder_in_packed)

            encoder_out, _ = nn.utils.rnn.pad_packed_sequence(
                encoder_out_tmp, batch_first=True
            )
        else:
            encoder_out, h = self.encoder(encoder_in)

        h = h.swapaxes(0, 1).reshape(batch_size, -1)

        return self.attention(
            history=encoder_out,
            context=h,
            mask=lengths_to_mask(lengths, encoder_out.shape[1]),
        )


class Decoder(jit.ScriptModule):
    def __init__(
        self,
        decoder_in_dim: int,
        decoder_dropout: float,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: Optional[Literal["tanh"]] = None,
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

        linear_arr: list[nn.Module] = [nn.Linear(hidden_dim, output_dim)]
        if activation == "tanh":
            print("Decoder: Tanh activation")
            linear_arr.append(nn.Tanh())

        self.linear = nn.Sequential(*linear_arr)

    def get_hidden(self, batch_size, device):
        return [
            torch.zeros((batch_size, self.hidden_dim), device=device)
            for x in range(self.num_layers)
        ]

    @jit.script_method
    def forward(
        self,
        encoded: Tensor,
        hidden: List[Tensor],
    ) -> Tuple[Tensor, List[Tensor]]:
        if len(hidden) != self.num_layers:
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        new_hidden: List[Tensor] = []

        x = encoded

        for i, rnn in enumerate(self.rnn):
            h = hidden[i]

            h_out = rnn(x, h)
            x = h_out

            if i < (len(self.rnn) - 1):
                x = self.dropout(x)

            new_hidden.append(h_out)

        x = self.linear(x)

        return x, new_hidden


class DualAttentionDecoder(nn.Module):
    def __init__(
        self,
        attention_in_dim: int,
        attention_context_dim: int,
        attention_dim: int,
        decoder_in_dim: int,
        decoder_dropout: float,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: Optional[Literal["tanh"]],
    ):
        super().__init__()

        self.num_layers = num_layers

        self.attention_ours: Attention = Attention(
            history_in_dim=attention_in_dim,
            context_dim=attention_context_dim,
            att_dim=attention_dim,
        )

        self.attention_theirs: Attention = Attention(
            history_in_dim=attention_in_dim,
            context_dim=attention_context_dim,
            att_dim=attention_dim,
        )

        self.rnn = nn.ModuleList(
            [nn.LSTMCell(decoder_in_dim, hidden_dim)]
            + [nn.LSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )

        self.dropout = nn.Dropout(decoder_dropout)

        linear_arr = [nn.Linear(hidden_dim, output_dim)]
        if activation == "tanh":
            print("Decoder: Tanh activation")
            linear_arr.append(nn.Tanh())

        self.linear = nn.Sequential(*linear_arr)

    def forward(
        self,
        encoded_seq_ours: Tensor,
        encoded_seq_ours_len: Tensor,
        encoded_seq_theirs: Tensor,
        encoded_seq_theirs_len: Tensor,
        attention_context: Tensor,
        additional_decoder_input: List[Tensor],
        hidden: List[Tuple[Tensor, Tensor]],
        hidden_idx: Tensor,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]], Tensor]:
        if len(hidden) != self.num_layers:
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        encoded_att_ours, scores_ours = self.attention_theirs(
            encoded_seq_theirs,
            attention_context,
            lengths_to_mask(encoded_seq_theirs_len, encoded_seq_theirs.shape[1]),
        )

        encoded_att_theirs, scores_theirs = self.attention_theirs(
            encoded_seq_ours,
            attention_context,
            lengths_to_mask(encoded_seq_ours_len, encoded_seq_ours.shape[1]),
        )

        x = torch.cat(
            [encoded_att_ours, encoded_att_theirs] + additional_decoder_input, dim=-1
        )
        new_hidden: List[Tuple[Tensor, Tensor]] = []

        for i, rnn in enumerate(self.rnn):
            h, c = hidden[i]

            h_out, c_out = rnn(x, (h[hidden_idx], c[hidden_idx]))
            x = h_out

            if i < (len(self.rnn) - 1):
                x = self.dropout(x)

            new_hidden.append(
                (
                    h.index_copy(0, hidden_idx, h_out.type(h.dtype)),
                    c.index_copy(0, hidden_idx, c_out.type(c.dtype)),
                )
            )
        x = self.linear(x)

        return x, new_hidden, scores_ours, scores_theirs


class ExternalAttentionDecoder(nn.Module):
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

        self.rnn = nn.ModuleList(
            [nn.LSTMCell(decoder_in_dim, hidden_dim)]
            + [nn.LSTMCell(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )

        self.dropout = nn.Dropout(decoder_dropout)

        linear_arr = [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        ]
        if activation == "tanh":
            print("Decoder: Tanh activation")
            linear_arr.append(nn.Tanh())

        self.linear = nn.Sequential(*linear_arr)

    def forward(
        self,
        encoded_att: Tensor,
        additional_decoder_input: List[Tensor],
        hidden: List[Tuple[Tensor, Tensor]],
        hidden_idx: Tensor,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]], Tensor]:
        if len(hidden) != self.num_layers:
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        x = torch.cat([encoded_att] + additional_decoder_input, dim=-1)
        new_hidden: List[Tuple[Tensor, Tensor]] = []

        for i, rnn in enumerate(self.rnn):
            h, c = hidden[i]

            h_out, c_out = rnn(x, (h[hidden_idx], c[hidden_idx]))
            x = h_out

            if i < (len(self.rnn) - 1):
                x = self.dropout(x)

            new_hidden.append(
                (
                    h.index_copy(0, hidden_idx, h_out.type(h.dtype)),
                    c.index_copy(0, hidden_idx, c_out.type(c.dtype)),
                )
            )
        x = self.linear(x)

        return x, new_hidden
