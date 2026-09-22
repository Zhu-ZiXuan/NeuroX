"""OTS threshold-selector model.

See Also:
    docs/reference/primitive/device/selector.md
"""

from __future__ import annotations

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, NonProfileModule, PolicyBase
from neurox.primitive.nonideality import apply_gaussian

__all__ = [
    "Selector",
    "SelectorConfig",
    "SelectorPolicy",
]


class SelectorConfig(ConfigBase):
    # === Threshold ===

    vth_nominal__V: float
    """Threshold voltage shared across all cells before mismatch."""

    # === Mismatch ===

    vth_mismatch__V: float
    """Standard deviation of the additive Gaussian threshold mismatch."""

    def validate(self) -> None:

        # --- Mismatch ---

        self._require_non_neg(self.vth_mismatch__V, "vth_mismatch__V")


class SelectorPolicy(PolicyBase):
    vth_mismatch: bool
    """Draw a per-cell threshold offset at fabricate time."""


_Config = SelectorConfig
_Policy = SelectorPolicy


class Selector(NonProfileModule):
    """OTS selector with static per-cell V_th mismatch."""

    config: _Config
    policy: _Policy

    # === Nominal buffers ===

    _nominal_vth__V: Tensor  # Shape: []

    # === Fabricated state ===

    _vth__V: Tensor  # Shape: [*inst_shape]

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._register_fabrication_buffers(dtype=dtype)

    def _register_fabrication_buffers(self, *, dtype: torch.dtype) -> None:
        self._register_nonpersistent_buffer(
            "_nominal_vth__V",
            torch.tensor(self.config.vth_nominal__V, dtype=dtype),
        )

    def _sample_fabrication_variation(self) -> None:
        self._vth__V = apply_gaussian(
            self._nominal_vth__V.clone().expand(self.inst_shape),
            sigma=self.config.vth_mismatch__V,
            enabled=self.policy.vth_mismatch,
        )

    @torch.no_grad()
    def sample_vth_like(self, reference: Tensor) -> Tensor:
        """Broadcast the fabricated threshold to the shape of `reference`.

        Returns:
            Per-cell threshold voltage [V].
        """
        return torch.broadcast_to(self._vth__V, reference.shape)
