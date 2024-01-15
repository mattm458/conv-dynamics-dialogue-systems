from typing import Final, Literal, Optional

import torch
from torch import Generator, Tensor

from cdmodel.consts import SPEAKER_ROLE_AGENT_IDX, SPEAKER_ROLE_PARTNER_IDX


def lengths_to_mask(lengths: Tensor, max_size: int) -> Tensor:
    mask = torch.arange(max_size, device=lengths.device)
    mask = mask.unsqueeze(0).repeat(len(lengths), 1)
    mask = mask >= lengths.unsqueeze(1)
    mask = mask.unsqueeze(2)
    return mask


def timestep_split(tensor: Tensor) -> list[Tensor]:
    return [x.squeeze(1) for x in torch.split(tensor, 1, dim=1)]


def get_role_identity_idx(
    speaker_identity_idx: Tensor,
    role_assignment: Literal["agent_first", "agent_second", "random"],
    zero_pad: bool,
    generator: Optional[Generator] = None,
) -> tuple[Tensor, Tensor]:
    """
    Given a tensor of speaker identity indices in a segmented conversation,
    assign roles ("agent" or "partner") to the speakers.

    Roles are assigned based on one of three strategies:

    * `agent_first`: The agent is the first person to speak in the conversation, and
      their partner is the second to speak.
    * `agent_second`: The agent is the second person to speak in the conversation, and
      their partner is the first to speak.
    * `random`: Agent and partner roles are assigned randomly.

    For random assignment, a PyTorch Generator object is required. This is to ensure
    that random numbers are assigned on the same device as the output tensor, and
    that situations requiring deterministic output (for example, random role selection
    during validation) can be set up in advance.

    Parameters
    ----------
    speaker_identity_idx : Tensor
        A tensor of batched segmented speaker identity indices. It must have the
        dimensions (batch, segments).
    role_assignment : Literal["agent_first", "agent_second", "random"]
        The method used to assign speaker roles.
    zero_pad : bool
        Whether the `speaker_identity_idx` tensor is zero-padded. If it is, the first
        index along the segment dimension is 0, which is not a valid speaker identity.
    generator : Optional[Generator], optional
        A PyTorch Generator for random numbers. Required if `role_assignment` is
        `random`. By default `None`.

    Returns
    -------
    tuple[Tensor, Tensor]
        A tuple containing two tensors: the speaker identity index of the agents in each
        batch, and the speaker identity index of the partners in each batch.

    Raises
    ------
    Exception
        Raised if the role assignment strategy is `random` but no Generatorwas given
        as an argument.
    NotImplementedError
        Raised if an unknown role assignment strategy is given as an argument.
    """
    batch_size: Final[int] = speaker_identity_idx.shape[0]

    # Establish the speaker role
    if role_assignment == "agent_first":
        agent_segment_idx: int = int(zero_pad)
        agent_identity_idx: Tensor = speaker_identity_idx[:, agent_segment_idx]
        partner_identity_idx: Tensor = speaker_identity_idx[:, agent_segment_idx + 1]
    elif role_assignment == "agent_second":
        agent_segment_idx = int(zero_pad) + 1
        agent_identity_idx = speaker_identity_idx[:, agent_segment_idx]
        partner_identity_idx = speaker_identity_idx[:, agent_segment_idx - 1]
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

        agent_identity_idx = torch.gather(
            speaker_identity_idx, 1, agent_segment_idxs.unsqueeze(1)
        ).squeeze(1)
        partner_identity_idx = torch.gather(
            speaker_identity_idx, 1, partner_segment_idxs.unsqueeze(1)
        ).squeeze(1)
    else:
        raise NotImplementedError(
            f'Role assignment method "{role_assignment}" not supported'
        )

    return agent_identity_idx, partner_identity_idx


def get_speaker_role_idx(
    speaker_identity_idx: Tensor,
    agent_identity_idx: Tensor,
    partner_identity_idx: Tensor,
) -> Tensor:
    """
    Generate a tensor of speaker role indices from a tensor of speaker identity indices,
    plus separate tensors indicating which speaker identities are assuming the role of agent
    and which speaker identities are assuming the role of partner.

    As an example, assume the speaker identity indices contain the following batched segments:

    ```py
    [
        [5, 6, 5, 6, 5, 0],
        [6, 7, 6, 7, 0, 0],
        [3, 4, 3, 4, 3, 3]
    ]
    ```

    If the agent identities are `[5, 6, 4]` and the partner identities are `[6, 7, 3]`,
    then the output of this function will be the following tensor:

    ```py
    [
        [2, 1, 2, 1, 2, 0],
        [2, 1, 2, 1, 0, 0],
        [1, 2, 1, 2, 1, 1]
    ]
    ```

    This is because the agent role index is hardcoded at 2, and the speaker role index at 1.
    Values of 0 are ignored, since they are used for padding.


    Parameters
    ----------
    speaker_identity_idx : Tensor
        A tensor of batched segmented speaker identity indices. It must have the
        dimensions (batch, segments).
    agent_identity_idx : Tensor
        The speaker identity indidices of the agents in each batch.
    partner_identity_idx : Tensor
        The speaker identity indidices of the partners in each batch.

    Returns
    -------
    Tensor
        A tensor of batched segmented speaker role indices with the dimensions (batch, segments).
    """
    is_agent: Final[Tensor] = speaker_identity_idx.eq(agent_identity_idx.unsqueeze(1))
    is_partner: Final[Tensor] = speaker_identity_idx.eq(
        partner_identity_idx.unsqueeze(1)
    )
    speaker_role_idx: Final[Tensor] = torch.zeros_like(speaker_identity_idx)
    speaker_role_idx[is_agent] = SPEAKER_ROLE_AGENT_IDX
    speaker_role_idx[is_partner] = SPEAKER_ROLE_PARTNER_IDX

    return speaker_role_idx
