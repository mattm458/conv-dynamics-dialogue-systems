from typing import Final, NamedTuple, Optional

import torch
from torch import Tensor


class AnchorTimestep(NamedTuple):
    predict_timestep: int
    scores: Tensor


class Anchor(NamedTuple):
    highest_scoring_segment: int
    timesteps: list[AnchorTimestep]


def find_anchors(scores_all: list[Tensor]) -> list[Anchor | None]:
    """
    Anchors are historical segments in a conversation that are scored the
    highest by an attention mechanism for several consecutive output timesteps.

    This function extracts anchors from all attention scores in a conversation.
    *Note that determining an appropriate length for anchors is outside the scope
    of this function*: it will return achors that are relevant for only one
    timestep, which may not be appropriate for an anchor analysis. You must
    filter the output of this function to remove anchors which are too short!


    Parameters
    ----------
    scores_all : list[Tensor]
        All attention scores from all historical conversation timesteps.

    Returns
    -------
    list[Anchor]
        All anchors found in the attention scores.
    """

    output: Final[list[Anchor | None]] = []
    current_highest_scoring_segment: Optional[int] = None
    current_anchor_timesteps: list[AnchorTimestep] = []

    for i, scores in enumerate(scores_all):
        if len(scores) < 1:
            output.append(None)
            continue

        this_highest_scoring_segment: int = int(torch.argmax(scores).item())

        # If the highest-scoring segment is different than the previous highest-scoring segment,
        # then we're starting a new anchor.
        if this_highest_scoring_segment != current_highest_scoring_segment:
            if current_highest_scoring_segment is not None:
                output.append(
                    Anchor(
                        highest_scoring_segment=current_highest_scoring_segment,
                        timesteps=current_anchor_timesteps,
                    )
                )

            current_highest_scoring_segment = this_highest_scoring_segment
            current_anchor_timesteps = []

        current_anchor_timesteps.append(
            AnchorTimestep(scores=scores, predict_timestep=i)
        )

    if (
        len(current_anchor_timesteps) > 0
        and current_highest_scoring_segment is not None
    ):
        output.append(
            Anchor(
                highest_scoring_segment=current_highest_scoring_segment,
                timesteps=current_anchor_timesteps,
            )
        )

    return output
