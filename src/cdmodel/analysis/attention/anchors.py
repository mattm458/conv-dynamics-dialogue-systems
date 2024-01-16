from typing import Final, NamedTuple

import torch
from torch import Tensor


class Anchor(NamedTuple):
    """
    An object representing a conversational anchor. An anchor is a segment
    that was found relevant by an attention mechanism for several consecutive
    output timesteps.

    Attributes
    ----------
    segment_idx : int
        The index of the historical segment.
    timesteps : list[int]
        The index of consecutive timesteps where `segment_idx` had the
        highest attention score.
    """

    segment_idx: int
    timesteps: list[int]


def get_att_anchors(att_scores: list[Tensor]) -> list[Anchor]:
    """
    Anchors are historical segments in a conversation that are scored the
    highest by an attention mechanism for several consecutive output timesteps.

    This function extracts anchors from all attention scores in a conversation.
    *Note that determining an appropriate length for anchors is outside the scope of this function*: it will return achors that are relevant for only one
    timestep, which may not be appropriate for an anchor analysis. You must
    filter the output of this function to remove anchors which are too short!


    Parameters
    ----------
    att_data : list[Tensor]
        A list of attention scores. The list contains attention scores at each
        output timestep in the conversaion.

    Returns
    -------
    list[Anchor]
        A list containing all Anchor objects found in the attention scores.
    """

    output: Final[list[Anchor]] = []

    anchor_this: list[int] = []
    prev_highest_score_idx: int = -1

    for i, att_timestep in enumerate(att_scores):
        highest_score_idx: int = int(torch.argmax(att_timestep))

        # Have we broken a streak of the same highest index?
        if highest_score_idx != prev_highest_score_idx:
            # If so, start a new streak at the current highest index

            # If the streak lasted longer than 1 timestep, save it
            if len(anchor_this) > 0:
                output.append(
                    Anchor(segment_idx=prev_highest_score_idx, timesteps=anchor_this)
                )
            anchor_this = [i]

            prev_highest_score_idx = int(highest_score_idx)

        # If not, save the current timestep index
        else:
            anchor_this.append(i)

    if len(anchor_this) > 0:
        output.append(Anchor(segment_idx=prev_highest_score_idx, timesteps=anchor_this))

    return output
