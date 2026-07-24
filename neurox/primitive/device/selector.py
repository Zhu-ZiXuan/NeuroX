"""OTS threshold-selector model.

See also:
    docs/reference/primitive/device/selector.md
"""

from typing import ClassVar

import torch
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.primitive.nonideality import apply_gaussian


class SelectorConfig(ConfigBase):
    """Immutable configuration for an OTS threshold selector.

    Attributes:
        vth_nominal__V: Nominal threshold voltage ``V_th`` shared
            across all cells before mismatch is applied.
        vth_mismatch__V: Additive Gaussian mismatch on ``V_th``.
    """

    vth_nominal__V: float

    vth_mismatch__V: float

    def validate(self) -> None:
        self._require_non_neg(self.vth_mismatch__V, "vth_mismatch__V")


class SelectorPolicy(PolicyBase):
    """Per-source toggles selecting which selector nonidealities are active.

    Attributes:
        vth_mismatch: Apply ``vth_mismatch__V`` per cell at fabricate time.
    """

    vth_mismatch: bool


class Selector(ModuleBase[SelectorConfig, SelectorPolicy]):
    """OTS selector with static per-cell V_th mismatch."""

    is_profile_target: ClassVar[bool] = False

    # --- Fabrication source buffers ---

    nominal_vth__V: Tensor

    def __init__(
        self,
        *,
        config: SelectorConfig,
        policy: SelectorPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)
        self._register_fabrication_buffers(dtype=dtype)

    def _register_fabrication_buffers(self, *, dtype: torch.dtype) -> None:
        """Register immutable tensors used as fabrication sources."""
        self.register_buffer(
            "nominal_vth__V",
            torch.tensor(self.config.vth_nominal__V, dtype=dtype),
            persistent=False,
        )

    def _sample_fabricate_mismatch(self) -> None:
        self.vth__V = apply_gaussian(
            self.nominal_vth__V.clone().expand(self.inst_shape),
            self.config.vth_mismatch__V,
            enabled=self.policy.vth_mismatch,
        )

    def sample_vth_like(self, reference: Tensor) -> Tensor:
        """Broadcast the fabricated ``vth__V`` to ``reference``'s shape/device/dtype.

        Args:
            reference: Tensor whose shape, device, and dtype define the
                target threshold tensor (typically the cell-voltage tensor).

        Returns:
            Threshold voltage tensor [V]. Shape: ``reference.shape``.
        """
        return torch.broadcast_to(self.vth__V, reference.shape)
