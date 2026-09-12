"""Element-wise integer subtractor.

See Also:
    docs/reference/primitive/digital/subtractor.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import DigitalBase, DigitalConfig, DigitalPolicy


class SubtractorConfig(DigitalConfig):
    bit_width: int
    """Nominal output bit width; sizes the PPA, no wrap is applied."""

    energy_per_op__fJ: float
    """Dynamic energy per output element."""
    latency_per_op__ns: float
    """Combinational window of one subtract."""

    def validate(self) -> None:
        super().validate()

        # --- Arithmetic ---

        self._require_pos(self.bit_width, "bit_width")

        # --- PPA ---

        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")
        self._require_non_neg(self.latency_per_op__ns, "latency_per_op__ns")


_Config = SubtractorConfig
_Policy = DigitalPolicy


class Subtractor(DigitalBase):
    """Element-wise integer subtractor without saturation or wrapping."""

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
        """Combinational latency of one subtract."""
        return self.config.latency_per_op__ns

    @torch.no_grad()
    def subtract(self, a: Tensor, b: Tensor) -> Tensor:
        """Subtract one integer tensor from another element-wise.

        The subtrahend `b` broadcasts against the minuend `a`.

        Returns:
            `a - b`, unwrapped and unsaturated.
        """
        y = a - b
        if self._is_dynamic_energy_profile_active():
            e_op__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32, device=y.device)
            self._record_dynamic_energy(e_op__fJ.expand(y.shape))
        return y
