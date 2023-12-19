import torch
from torch import Tensor


def lengths_to_mask(lengths: Tensor, max_size: int) -> Tensor:
    mask = torch.arange(max_size, device=lengths.device)
    mask = mask.unsqueeze(0).repeat(len(lengths), 1)
    mask = mask >= lengths.unsqueeze(1)
    mask = mask.unsqueeze(2)
    return mask

def timestep_split(tensor: Tensor) -> list[Tensor]:
    return [x.squeeze(1) for x in torch.split(tensor, 1, dim=1)]
