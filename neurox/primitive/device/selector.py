"""OTS threshold-selector model.

See Also:
    docs/reference/primitive/device/selector.md
"""

from typing import ClassVar

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.primitive.nonideality import apply_gaussian


class SelectorConfig(ConfigBase):
    """Immutable configuration for an OTS threshold selector."""

    vth_nominal__V: float
    """Threshold voltage shared across all cells before mismatch."""

    vth_mismatch__V: float
    """Standard deviation of the additive Gaussian threshold mismatch."""

    def validate(self) -> None:
        self._require_non_neg(self.vth_mismatch__V, "vth_mismatch__V")


class SelectorPolicy(PolicyBase):
    """Per-source toggles selecting which selector nonidealities are active."""

    vth_mismatch: bool
    """Draw a per-cell threshold offset at fabricate time."""


class Selector(ModuleBase[SelectorConfig, SelectorPolicy]):
    """OTS selector with static per-cell V_th mismatch."""

    is_profile_target: ClassVar[bool] = False

    # === Nominal buffers ===

    _nominal_vth__V: Tensor  # Shape: []

    # === Fabricated state ===

    _vth__V: Tensor  # Shape: [*inst_shape]

    def __init__(
        self,
        *,
        config: SelectorConfig,
        policy: SelectorPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        # `T__K` completes the uniform device-construction signature; this threshold model
        # has no temperature dependence to apply it to.
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._register_fabrication_buffers(dtype=dtype)

    def _register_fabrication_buffers(self, *, dtype: torch.dtype) -> None:
        self.register_buffer(
            "_nominal_vth__V",
            torch.tensor(self.config.vth_nominal__V, dtype=dtype),
            persistent=False,
        )

    def _sample_fabrication_variation(self) -> None:
        self._vth__V = apply_gaussian(
            self._nominal_vth__V.clone().expand(self.inst_shape),
            self.config.vth_mismatch__V,
            enabled=self.policy.vth_mismatch,
        )

    def sample_vth_like(self, reference: Tensor) -> Tensor:
        """Broadcast the fabricated threshold to the shape of `reference`.

        Returns:
            Per-cell threshold voltage [V].
            Shape: `[...]`.
        """
        return torch.broadcast_to(self._vth__V, reference.shape)
