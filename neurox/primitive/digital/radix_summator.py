"""Parallel positional summation for integer digit reconstruction.

See Also:
    docs/reference/primitive/digital.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import DigitalBase, DigitalConfig, DigitalPolicy


class RadixSummatorConfig(DigitalConfig):
    # === Dynamic energy ===

    energy_per_op__fJ: float
    """Dynamic energy per digit leg of one output."""

    def validate(self) -> None:
        super().validate()
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")


_Config = RadixSummatorConfig
_Policy = DigitalPolicy


class RadixSummator(DigitalBase):
    """Parallel positional-sum tree with signed output-width wrapping.

    The caller supplies PPA characterizing the selected digit-axis extent at the
    configured clock period.

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
    def radix_sum(self, x: Tensor, *, dim: int, radix: int, enable: Tensor | None = None) -> Tensor:
        """Reconstruct positional digits with signed output-width wrap.

        Args:
            x: Integer digits in least-significant-first order along `dim`.
                Shape: `[..., digit, ...]`.
            dim: Operand axis to reduce; negative indices count from the end.
            radix: Positional radix of this operation; 1 selects equal-weight
                summation.
            enable: Optional digit enables, broadcastable to `x`; disabled
                digits contribute zero and incur no evaluation energy.

        Returns:
            Reconstructed sum with `dim` removed, preserving the input dtype.
        """
        if radix < 1:
            raise ValueError(f"require: radix ({radix}) >= 1")
        if enable is not None:
            x = x.where(enable, 0)
        # Static radix and shape specialize the reduction without a device weight tensor.
        if radix == 1 or x.shape[dim] == 0:
            y = x.sum(dim=dim, dtype=x.dtype)
        else:
            parts = x.unbind(dim=dim)
            y = parts[-1]
            for part in reversed(parts[:-1]):
                y = y * radix + part
        y = self._wrap_output(y)
        if self._is_profiler_active():
            # Mask presence specializes during tracing; only masked costs depend on its device.
            if enable is None:
                energy__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32)
            else:
                energy__fJ = enable.to(dtype=torch.float32) * self.config.energy_per_op__fJ
            self._record_dynamic_energy(energy__fJ.expand(x.shape))
        return y
