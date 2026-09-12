"""Active-position layout for macro rescale fitting."""

from __future__ import annotations

import torch
from torch import Tensor


def unroll_active_positions(
    x: Tensor,
    *,
    input_num: int,
    max_active_num: int,
    inst_shape: tuple[int, ...],
) -> Tensor:
    """Partition inputs into planes aligned to the called macro's instances."""
    inst_rank = len(inst_shape)
    plane_num = -(-input_num // max_active_num)
    plane_of_input = torch.arange(input_num, device=x.device) // max_active_num
    mask = plane_of_input == torch.arange(plane_num, device=x.device).unsqueeze(-1)
    # Shape: [P, input] -> [P, *inst_shape=1, input]
    mask_shape = (mask.shape[0], *(1,) * inst_rank, mask.shape[-1])
    mask = mask.view(mask_shape)
    planes = torch.where(mask, x.unsqueeze(-(inst_rank + 2)), x.new_zeros(()))
    return planes.expand(*planes.shape[: -(inst_rank + 1)], *inst_shape, input_num)
