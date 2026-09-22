"""Active-position layout for macro rescale fitting."""

from __future__ import annotations

from torch import Tensor

from neurox.architecture.mapping import InputPhaseSplitter


def unroll_active_positions(
    x: Tensor,
    *,
    input_num: int,
    max_active_num: int,
    inst_shape: tuple[int, ...],
) -> Tensor:
    """Partition inputs into planes aligned to the called macro's instances."""
    inst_rank = len(inst_shape)
    splitter = InputPhaseSplitter(input_num=input_num, max_active_num=max_active_num)
    # Shape: [..., *inst_shape, input] -> [..., *inst_shape, input_phase, input]
    planes = splitter.split(x)
    # Shape: [..., *inst_shape, input_phase, input] -> [..., input_phase, *inst_shape, input]
    planes = planes.movedim(-2, -(inst_rank + 2))
    return planes.expand(*planes.shape[: -(inst_rank + 1)], *inst_shape, input_num)
