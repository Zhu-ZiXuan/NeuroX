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

    The caller supplies PPA characterizing the selected digit-axis extent at
    the configured clock period.
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
    def radix_sum(self, x: Tensor, dim: int, *, radix: int, enable: Tensor | None = None) -> Tensor:
        """Reconstruct positional digits with signed output-width wrap.

        Args:
            x: Integer digits in least-significant-first order along `dim`.
                Shape: `[..., digit, ...]`.
            radix: Positional radix of this operation; 1 selects equal-weight summation.
            enable: Optional digit enables, broadcastable to `x`; disabled
                digits contribute zero and incur no evaluation energy.

        Returns:
            Reconstructed sum with `dim` removed, preserving the input dtype.
        """
        if radix < 1:
            raise ValueError(f"require: radix ({radix}) >= 1")
        if enable is not None:
            x = x.where(enable, 0)
        # Shape: [digit]
        scales = x.new_tensor([radix**i for i in range(x.shape[dim])])
        # Shape: [..., digit, ...] -> [..., digit] -> [...]
        y = (x.movedim(dim, -1) * scales).sum(dim=-1, dtype=x.dtype)
        y = self._wrap_output(y)
        if self._is_profiler_active():
            e_op__fJ = torch.full((), self.config.energy_per_op__fJ, dtype=torch.float32, device=x.device)
            energy__fJ = e_op__fJ.expand(x.shape)
            if enable is not None:
                energy__fJ = energy__fJ.where(enable, 0)
            self._record_dynamic_energy(energy__fJ)
        return y
