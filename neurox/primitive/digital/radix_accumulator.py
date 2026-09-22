"""Modular accumulation over a positional digit axis.

See Also:
    docs/reference/primitive/digital.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import DigitalBase, DigitalConfig, DigitalPolicy


class RadixAccumulatorConfig(DigitalConfig):
    # === Dynamic energy ===

    energy_per_op__fJ: float
    """Dynamic energy per enabled accumulator update."""

    def validate(self) -> None:
        super().validate()
        self._require_non_neg(self.energy_per_op__fJ, "energy_per_op__fJ")


_Config = RadixAccumulatorConfig
_Policy = DigitalPolicy


class RadixAccumulator(DigitalBase):
    """One feedback register accumulating every digit on the selected axis."""

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
    def radix_accumulate(self, x: Tensor, dim: int, *, radix: int, enable: Tensor | None = None) -> Tensor:
        """Accumulate a complete positional stream with signed register-width wrap.

        Each output starts from zero. Disabled arrivals contribute zero and
        incur no update energy.

        Args:
            x: Integer digits in least-significant-first order along `dim`.
                Shape: `[..., digit, ...]`.
            radix: Positional radix of this operation; 1 selects equal-weight accumulation.
            enable: Optional arrival enables, broadcastable to `x`.

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
