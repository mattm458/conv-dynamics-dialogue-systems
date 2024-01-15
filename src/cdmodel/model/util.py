from typing import Final, Literal, Optional

import torch
from torch import Generator, Tensor


def lengths_to_mask(lengths: Tensor, max_size: int) -> Tensor:
    mask = torch.arange(max_size, device=lengths.device)
    mask = mask.unsqueeze(0).repeat(len(lengths), 1)
    mask = mask >= lengths.unsqueeze(1)
    mask = mask.unsqueeze(2)
    return mask


def timestep_split(tensor: Tensor) -> list[Tensor]:
    return [x.squeeze(1) for x in torch.split(tensor, 1, dim=1)]


def get_role_idx(
    speaker_id_idx: Tensor,
    role_assignment: Literal["first", "second", "random"],
    zero_pad: bool,
    generator: Optional[Generator] = None,
) -> tuple[Tensor, Tensor]:
    batch_size: Final[int] = speaker_id_idx.shape[0]

    # Establish the speaker role
    if role_assignment == "first":
        agent_segment_idx: int = int(zero_pad)
        agent_idx: Tensor = speaker_id_idx[:, agent_segment_idx]
        partner_idx: Tensor = speaker_id_idx[:, agent_segment_idx + 1]
    elif role_assignment == "second":
        agent_segment_idx = int(zero_pad) + 1
        agent_idx = speaker_id_idx[:, agent_segment_idx]
        partner_idx = speaker_id_idx[:, agent_segment_idx - 1]
    elif role_assignment == "random":
        if generator is None:
            raise Exception(
                "If using random role assignment, a Generator object must be given"
            )

        agent_idx_bool: Final[Tensor] = (
            torch.rand(batch_size, generator=generator, device=generator.device) < 0.5
        )
        partner_idx_bool: Final[Tensor] = ~agent_idx_bool

        agent_segment_idxs: Tensor = agent_idx_bool.type(torch.long)
        partner_segment_idxs: Tensor = partner_idx_bool.type(torch.long)

        if zero_pad:
            agent_segment_idxs += 1
            partner_segment_idxs += 1

        agent_idx = torch.gather(
            speaker_id_idx, 1, agent_segment_idxs.unsqueeze(1)
        ).squeeze(1)
        partner_idx = torch.gather(
            speaker_id_idx, 1, partner_segment_idxs.unsqueeze(1)
        ).squeeze(1)
    else:
        raise NotImplementedError(
            f'Role assignment method "{role_assignment}" not supported'
        )

    return agent_idx, partner_idx
