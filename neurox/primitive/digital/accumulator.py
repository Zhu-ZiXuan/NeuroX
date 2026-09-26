"""Modular accumulation over a temporal operand axis.

See Also:
    docs/reference/primitive/digital.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import DigitalBase, DigitalConfig, DigitalPolicy


class AccumulatorConfig(DigitalConfig):
    # === Dynamic energy ===

    energy_per_op__fJ: float
    """Dynamic energy per operand element folded into the sum."""

    def validate(self) -> None:
        super().validate()
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")


_Config = AccumulatorConfig
_Policy = DigitalPolicy


class Accumulator(DigitalBase):
    """One feedback register updated for every operand on the selected axis.

    Each method call reduces the complete supplied operand sequence from a fresh
    zero value; no accumulator state persists between calls. Supply an integer
    dtype wide enough for intermediate arithmetic and the configured wrap
    constants. The returned value wraps to the configured signed bit width and
    keeps the input dtype and device. No fabrication or programming is required.

    The caller supplies physical-instance and operation extents in the operands;
    `inst_shape` scales static cost but does not broadcast extra dynamic
    operations. The containing unit owns scheduling and elapsed time.

    Args:
        config: Hardware configuration.
        policy: Run policy matching `config`.
        inst_shape: Positive physical instance extents; singletons allow
            broadcasting.
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
    def accumulate(self, x: Tensor, *, dim: int, enable: Tensor | None = None) -> Tensor:
        """Reduce a complete arrival stream with signed register-width wrap.

        Each output starts from zero. Disabled arrivals contribute zero and
        incur no update energy.

        Args:
            x: Integer operands in arrival order along `dim`.
                Shape: `[..., operand, ...]`.
            dim: Operand axis to reduce; negative indices count from the end.
            enable: Optional arrival enables, broadcastable to `x`.

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
