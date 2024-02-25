from typing import Literal, NamedTuple

import torch
from torch import Tensor, jit, nn

from cdmodel.model.util import lengths_to_mask


class Encoder(jit.ScriptModule):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.ModuleList(
            [nn.GRUCell(in_dim, hidden_dim)]
            + [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )

        self.dropout = nn.Dropout(dropout)

    def get_hidden(self, batch_size: int, device) -> list[Tensor]:
        return [
            torch.zeros((batch_size, self.hidden_dim), device=device)
            for x in range(self.num_layers)
        ]

    @jit.script_method
    def forward(
        self,
        encoder_input: Tensor,
        hidden: list[Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        if len(hidden) != len(self.rnn):
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        x = encoder_input
        new_hidden: list[Tensor] = []

        for i, rnn in enumerate(self.rnn):
            h_out = rnn(x, hidden[i])
            x = h_out
            new_hidden.append(h_out)

            if i < (len(self.rnn) - 1):
                x = self.dropout(x)

        return x, new_hidden


class AttentionModule(jit.ScriptModule):
    pass


class AttentionScores(NamedTuple):
    agent_scores: Tensor | None = None
    partner_scores: Tensor | None = None
    combined_scores: Tensor | None = None


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
    ) -> tuple[Tensor, Tensor]:
        history_att: Tensor = self.history(history)
        context_att: Tensor = self.context(context).unsqueeze(1)

        score: Tensor = self.v(torch.tanh(history_att + context_att))
        score = score.masked_fill(mask, float("-inf"))
        score = torch.softmax(score, dim=1)
        score = score.masked_fill(mask, 0.0)
        score = score.swapaxes(1,2)

        att_applied = torch.bmm(score, history).squeeze(1)

        return att_applied, score.detach().clone()


class DualAttention(AttentionModule):
    def __init__(self, history_in_dim: int, context_dim: int, att_dim: int):
        super().__init__()

        self.our_attention = Attention(
            history_in_dim=history_in_dim,
            context_dim=context_dim,
            att_dim=att_dim,
        )
        self.their_attention = Attention(
            history_in_dim=history_in_dim,
            context_dim=context_dim,
            att_dim=att_dim,
        )

    @jit.script_method
    def forward(
        self, history: Tensor, context: Tensor, agent_mask: Tensor, partner_mask: Tensor
    ) -> tuple[Tensor, AttentionScores]:
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

        return torch.cat([agent_att, partner_att], dim=-1), AttentionScores(
            agent_scores=agent_scores,
            partner_scores=partner_scores,
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
    ) -> tuple[Tensor, AttentionScores]:
        att, scores = self.attention(
            history,
            context=context,
            mask=~(agent_mask + partner_mask).unsqueeze(-1),
        )

        return att, AttentionScores(combined_scores=scores)


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
    ) -> tuple[Tensor, AttentionScores]:
        att, scores = self.attention(
            history,
            context=context,
            mask=~(partner_mask).unsqueeze(-1),
        )

        return att, AttentionScores(partner_scores=scores)


class NoopAttention(AttentionModule):
    @jit.script_method
    def forward(
        self, history: Tensor, context: Tensor, agent_mask: Tensor, partner_mask: Tensor
    ) -> tuple[Tensor, tuple[Tensor | None, Tensor | None]]:
        return history[:, -1], (None, None)


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
    def forward(self, encoder_in: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
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
        activation: Literal["tanh", None],
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

    def get_hidden(self, batch_size: int, device) -> list[Tensor]:
        return [
            torch.zeros((batch_size, self.hidden_dim), device=device)
            for x in range(self.num_layers)
        ]

    @jit.script_method
    def forward(
        self,
        encoded: Tensor,
        hidden: list[Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        if len(hidden) != self.num_layers:
            raise Exception(
                "Number of hidden tensors must equal the number of RNN layers!"
            )

        new_hidden: list[Tensor] = []

        x = encoded

        for i, rnn in enumerate(self.rnn):
            h_out = rnn(x, hidden[i])
            x = h_out

            if i < (len(self.rnn) - 1):
                x = self.dropout(x)

            new_hidden.append(h_out)

        x = self.linear(x)

        return x, new_hidden
