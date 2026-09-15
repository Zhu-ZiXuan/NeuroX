"""Input-axis grouping and recovery for bounded simultaneous activation."""

from __future__ import annotations

import torch
from torch import Tensor


class InputActivation:
    """Select consecutive input groups in separate phases.

    Args:
        input_num: Width of the local input vector.
        max_active_num: Maximum number of input positions selected in one phase.
    """

    def __init__(self, *, input_num: int, max_active_num: int) -> None:
        self.input_num = input_num
        self.max_active_num = max_active_num
        self.input_phase_num = -(-input_num // max_active_num)

    def map_x(self, x: Tensor) -> Tensor:
        """Retain each input position in exactly one phase, zeroing the others.

        Args:
            x: Local input vectors; leading axes pass through unchanged.
                Shape: `[..., input]`.

        Returns:
            Phase inputs with at most `max_active_num` selected positions each.
            Shape: `[..., input_phase, input]`.
        """
        # Shape: [input]
        input_phase = torch.arange(self.input_num, device=x.device) // self.max_active_num
        # Shape: [input_phase, input=1]
        phases = torch.arange(self.input_phase_num, device=x.device).unsqueeze(-1)
        # Shape: [input_phase, input]
        mask = input_phase == phases
        # Shape: [..., input] -> [..., input_phase, input]
        return x.unsqueeze(-2).where(mask, 0)

    def recover(self, values: Tensor, *, dim: int) -> Tensor:
        """Sum phase results without hardware register-width wrap or circuit cost.

        Args:
            values: Partial results for each input phase.
                Shape: `[..., input_phase, ...]`.
            dim: Axis indexing the input phases.

        Returns:
            Sum with the phase axis removed, preserving the dtype.
        """
        # Shape: [..., input_phase, ...] -> [..., ...]
        return values.sum(dim=dim, dtype=values.dtype)
