"""Two-input addition with signed output-width wrapping.

See Also:
    docs/reference/primitive/digital.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import DigitalBase, DigitalConfig, DigitalPolicy


class AdderConfig(DigitalConfig):
    # === Dynamic energy ===

    energy_per_op__fJ: float
    """Dynamic energy per enabled two-input addition."""

    def validate(self) -> None:
        super().validate()
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")


_Config = AdderConfig
_Policy = DigitalPolicy


class Adder(DigitalBase):
    """Two-input adder with signed output-width wrapping.

    Inputs share an integer dtype and device and broadcast to the result shape.
    An optional enable mask broadcasts to that shape without enlarging it.
    Disabled operations return zero and incur no evaluation energy. Each enabled
    result incurs one operation's energy, including a zero-valued result.
    """

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

    @torch.no_grad()
    def add(self, a: Tensor, b: Tensor, *, enable: Tensor | None = None) -> Tensor:
        """Add two operands, applying enable gating and signed output-width wrap."""
        result = self._wrap_output(a + b)
        if enable is not None:
            enable = torch.broadcast_to(enable, result.shape)
            result = result.where(enable, 0)
        if self._is_profiler_active():
            e_op__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32, device=result.device)
            energy__fJ = e_op__fJ.expand(result.shape)
            if enable is not None:
                energy__fJ = energy__fJ.where(enable, 0)
            self._record_dynamic_energy(energy__fJ)
        return result
