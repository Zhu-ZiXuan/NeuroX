"""OTS threshold-selector model.

See also:
    docs/reference/device/selector.md
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.mixin import FabricateMixin, ValidateMixin
from neurox.common.nonideality import apply_gaussian


@dataclass(frozen=True)
class SelectorConfig(ValidateMixin):
    """Immutable configuration for an OTS threshold selector.

    Attributes:
        vth_nominal__V: Nominal threshold voltage ``V_th`` shared
            across all cells before mismatch is applied.
        vth_mismatch__V: Additive Gaussian mismatch on ``V_th``.
    """

    # --- Nominal threshold ---
    vth_nominal__V: float

    # --- V_th mismatch ---
    vth_mismatch__V: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._require_nonneg(self.vth_mismatch__V, "vth_mismatch__V")


@dataclass(frozen=True)
class SelectorPolicy:
    """Per-source toggles selecting which selector nonidealities are active.

    Attributes:
        vth_mismatch: Apply ``vth_mismatch__V`` per cell at fabricate time.
    """

    vth_mismatch: bool


class Selector(FabricateMixin, nn.Module):
    """OTS selector with static per-cell V_th mismatch."""

    nominal_vth__V: Tensor
    vth__V: Tensor

    def __init__(
        self,
        *,
        config: SelectorConfig,
        policy: SelectorPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        """Initialize the selector model."""
        super().__init__()
        self.config = config
        self.policy = policy
        self._inst_shape = inst_shape
        self.T__K = T__K
        self.dtype = dtype
        self.register_buffer(
            "nominal_vth__V",
            torch.tensor(config.vth_nominal__V, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "vth__V",
            self.nominal_vth__V.clone(),
            persistent=False,
        )

    def _sample_fabricate_mismatch(self) -> None:
        """Resample per-cell V_th at ``self._inst_shape``."""
        self.vth__V = apply_gaussian(
            self.nominal_vth__V.clone().expand(self._inst_shape),
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
        vth = self.vth__V.to(device=reference.device, dtype=reference.dtype)
        return torch.broadcast_to(vth, reference.shape)
