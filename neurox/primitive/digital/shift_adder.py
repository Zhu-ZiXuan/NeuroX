"""Shift-adder for multi-digit partial-product recombination.

See Also:
    docs/reference/primitive/digital/shift_adder.md
"""

import torch
from torch import Tensor

from .base import DigitalBase, DigitalConfig, DigitalPolicy


class ShiftAdderConfig(DigitalConfig):
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

        One operation is one shift-and-add cell per digit leg folded in; a
        preload of the destination register adds no evaluation of its own.

        Args:
            x: Integer digit tensor.
                Shape: `[..., digit_count, ...]`.
            dim: Axis indexing the digit positions.
            init_val: Optional partial sum added after the modular wrap,
                broadcastable to the reduced output shape.

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
            # Added after the wrap so a chained running total survives past one call's
            # register range; folded in before it, the total would be clipped each call.
            y = y + init_val

        if self._is_dynamic_energy_profile_active():
            # Shape: [] -> [*x.shape]
            e_op__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32, device=x.device)
            self._record_dynamic_energy(e_op__fJ.expand(x.shape))
        return y
