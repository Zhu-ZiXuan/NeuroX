"""Input-axis grouping and recovery for bounded simultaneous activation."""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.primitive.digital import Accumulator, RadixAccumulator


class InputPhaseSplitter:
    """Split consecutive input positions evenly across the minimum number of phases.

    Construction binds an ordinary reference to the owner's `accumulator`.
    A radix accumulator performs the phase sum with radix one.
    An absent reference selects exact functional summation without circuit cost.
    Input selection is fixed at construction and materialized on the input's device.

    Each phase selects at most `max_active_num` of the `input_num` positions.
    """

    def __init__(
        self,
        *,
        input_num: int,
        max_active_num: int,
        accumulator: Accumulator | RadixAccumulator | None = None,
    ) -> None:
        self.accumulator = accumulator
        self.input_num = input_num
        self.max_active_num = max_active_num
        self.input_phase_num = -(-input_num // max_active_num)
        self._mask = self._make_phase_mask()

    def split(self, x: Tensor, *, dim: int = -2) -> Tensor:
        """Retain each input position in exactly one phase, zeroing the others.

        Group sizes differ by at most one; earlier phases receive the extra
        positions. Selected positions retain their original input indices.

        Args:
            x: Local input vectors; leading axes pass through unchanged.
                Shape: `[..., input]`.
            dim: Axis at which the input-phase dimension is inserted.

        Returns:
            Phase inputs with at most `max_active_num` selected positions each,
            carrying a new input-phase axis at `dim`.
            Shape: `[..., input_phase, ..., input]`.
        """
        # Shape: [input_phase, input]
        mask = x.new_tensor(self._mask, dtype=torch.bool)
        # Shape: [..., input] -> [..., input_phase, ..., input]
        return x.unsqueeze(-2).where(mask, 0).movedim(-2, dim)

    def recover(self, values: Tensor, *, dim: int, enable: Tensor | None = None) -> Tensor:
        """Sum phase results through the bound accumulator or functional path.

        Args:
            values: Partial results for each input phase.
                Shape: `[..., input_phase, ...]`.
            dim: Axis indexing the input phases.
            enable: Optional operand enables, broadcastable to `values`.

        Returns:
            Sum with the phase axis removed, preserving the dtype.
        """
        # Circuit type is fixed at construction, so dispatch resolves during tracing.
        if isinstance(self.accumulator, RadixAccumulator):
            return self.accumulator.radix_accumulate(values, dim=dim, radix=1, enable=enable)
        if isinstance(self.accumulator, Accumulator):
            return self.accumulator.accumulate(values, dim=dim, enable=enable)
        if enable is not None:
            values = values.where(enable, 0)
        # Shape: [..., input_phase, ...] -> [..., ...]
        return values.sum(dim=dim, dtype=values.dtype)

    def _make_phase_mask(self) -> tuple[tuple[bool, ...], ...]:
        group_size, extra = divmod(self.input_num, self.input_phase_num)
        masks: list[tuple[bool, ...]] = []
        start = 0
        for phase in range(self.input_phase_num):
            size = group_size + (phase < extra)
            masks.append((False,) * start + (True,) * size + (False,) * (self.input_num - start - size))
            start += size
        return tuple(masks)
