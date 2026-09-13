"""Digital modular-arithmetic accumulator over an integer-tensor axis.

See Also:
    docs/reference/primitive/digital/accumulator.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import DigitalBase, DigitalConfig, DigitalPolicy


class AccumulatorConfig(DigitalConfig):
    bit_width: int
    """Signed output bit width; the result wraps modulo `2^bit_width` into
    `[-2^(bit_width-1), 2^(bit_width-1) - 1]`."""

    energy_per_op__fJ: float
    """Dynamic energy per operand element folded into the sum."""
    latency_per_op__ns: float
    """Reduction window of one accumulate."""

    def validate(self) -> None:
        super().validate()

        # --- Arithmetic ---

        self._require_pos(self.bit_width, "bit_width")

        # --- PPA ---

        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


_Config = AccumulatorConfig
_Policy = DigitalPolicy


class Accumulator(DigitalBase):
    """Modular adder-tree that sums an integer tensor along one axis."""

    config: _Config
    policy: _Policy

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    @property
    def _area_per_inst__um2(self) -> float:
        return self.config.area_per_inst__um2

    @property
    def _leakage_per_inst__uW(self) -> float:
        return self.config.leakage_per_inst__uW

    def latency__ns(self) -> float:
        """Latency of one accumulator evaluation."""
        return self.config.latency_per_op__ns

    @torch.no_grad()
    def accumulate(self, x: Tensor, dim: int) -> Tensor:
        """Sum integer `x` along `dim` and wrap into the signed `bit_width` range.

        One operation is one adder evaluation per operand element folded in.

        Returns:
            Modular-wrapped sum with `dim` reduced, preserving the input dtype.
        """
        bw = self.config.bit_width
        half = 1 << (bw - 1)
        full = 1 << bw
        y = (x.sum(dim, dtype=x.dtype) + half) % full - half

        if self._is_dynamic_energy_profile_active():
            e_op__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32, device=x.device)
            self._record_dynamic_energy(e_op__fJ.expand(x.shape))
        return y
