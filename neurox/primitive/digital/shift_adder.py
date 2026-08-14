"""Shift-adder for multi-digit partial-product recombination.

See Also:
    docs/reference/primitive/digital/shift_adder.md
    docs/internals/primitive/digital/shift_adder.md
"""

import torch
from torch import Tensor

from .base import DigitalBase, DigitalConfig, DigitalPolicy


class ShiftAdderConfig(DigitalConfig):
    """Immutable configuration for a ShiftAdder instance."""

    bit_width: int
    """Signed output bit width; the result wraps modulo `2^bit_width` into
    `[-2^(bit_width-1), 2^(bit_width-1) - 1]`."""

    energy_per_op__fJ: float
    """Dynamic energy per digit leg of one output."""
    latency_per_op__ns: float
    """Positional-sum window of one shift-add."""

    def validate(self) -> None:
        super().validate()

        # --- Arithmetic ---

        self._require_pos(self.bit_width, "bit_width")

        # --- PPA ---

        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


class ShiftAdder(DigitalBase[ShiftAdderConfig]):
    """Weighted positional-sum unit for digit recombination.

    Args:
        config: Arithmetic width and per-op PPA.
        policy: Empty digital policy marker.
        inst_shape: Per-instance fabrication multiplicity.
        scale: Positional radix; at least 2.
        digit_count: Number of positional digits reduced per operation.
    """

    # === Functional buffers ===

    _scales: Tensor  # Shape: [digit_count]

    def __init__(
        self,
        *,
        config: ShiftAdderConfig,
        policy: DigitalPolicy,
        inst_shape: tuple[int, ...],
        scale: int,
        digit_count: int,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        if scale < 2:
            raise ValueError(f"require: scale ({scale}) >= 2")
        if digit_count < 1:
            raise ValueError(f"require: digit_count ({digit_count}) >= 1")
        self.register_buffer(
            "_scales",
            torch.tensor([scale**i for i in range(digit_count)], dtype=torch.int64),
            persistent=False,
        )

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def shift_add(self, x: Tensor, dim: int, init_val: Tensor | None) -> Tensor:
        """Compute the radix-weighted digit sum and wrap to `bit_width` bits.

        Dynamic energy is billed against the pre-reduction operand: one
        shift-and-add cell per digit leg folded in, an extent the result no
        longer carries. A preload of the destination register adds no
        evaluation of its own.

        Args:
            x: Integer digit tensor.
                Shape: `[..., digit_count, ...]`.
            dim: Axis indexing the digit positions.
            init_val: Optional partial sum added after the modular wrap,
                broadcastable to the output shape.

        Returns:
            Recombined sum with `dim` reduced.
        """
        bw = self.config.bit_width
        half = 1 << (bw - 1)
        full = 1 << bw

        shape = [1] * x.ndim
        shape[dim] = x.size(dim)
        y = ((x * self._scales.view(*shape)).sum(dim=dim) + half) % full - half

        if init_val is not None:
            y = y + init_val

        if self._is_dynamic_energy_profile_active():
            # The engine positions this block's own inst_shape space axes
            # inside x, split from the batch by the reduced digit axis and any
            # further engine axes rather than held as one leading block;
            # billing the full pre-reduction operand covers them along with
            # the rest. A flat per-op lump: the expanded constant holds no
            # storage, and the energy dtype is the constant's rather than the
            # integer operand's.
            # Shape: [] -> [*x.shape]
            e_op__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32, device=x.device)
            self._record_dynamic_energy(e_op__fJ.expand(x.shape))
        return y
