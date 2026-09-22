"""Parallel modular summation over an operand axis.

See Also:
    docs/reference/primitive/digital.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import DigitalBase, DigitalConfig, DigitalPolicy


class SummatorConfig(DigitalConfig):
    # === Dynamic energy ===

    energy_per_op__fJ: float
    """Dynamic energy per operand element folded into the sum."""

    def validate(self) -> None:
        super().validate()
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")


_Config = SummatorConfig
_Policy = DigitalPolicy


class Summator(DigitalBase):
    """Parallel adder tree with signed output-width wrapping.

    The caller supplies PPA characterizing the selected operand-axis extent at
    the configured width and caller's clock period.
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
    def sum(self, x: Tensor, *, dim: int, enable: Tensor | None = None) -> Tensor:
        """Sum parallel operands with signed output-width wrap.

        Disabled inputs contribute zero and incur no evaluation energy.

        Args:
            x: Integer operands along `dim`.
                Shape: `[..., operand, ...]`.
            enable: Optional operand enables, broadcastable to `x`.

        Returns:
            Wrapped sum with `dim` removed, preserving the input dtype.
        """
        if enable is not None:
            x = x.where(enable, 0)
        # Shape: [..., operand, ...] -> [...]
        y = x.sum(dim, dtype=x.dtype)
        y = self._wrap_output(y)

        if self._is_profiler_active():
            # Mask presence specializes during tracing; only masked costs depend on its device.
            if enable is None:
                energy__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32)
            else:
                energy__fJ = enable.to(dtype=torch.float32) * self.config.energy_per_op__fJ
            self._record_dynamic_energy(energy__fJ.expand(x.shape))
        return y
