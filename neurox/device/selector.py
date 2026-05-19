"""Threshold-selector mismatch model."""

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.nonideality import apply_gaussian
from neurox.common.validate import ValidateMixin


@dataclass(frozen=True)
class SelectorConfig(ValidateMixin):
    """Immutable configuration for an OTS threshold selector.

    Attributes:
        vth_nominal__V: Nominal threshold voltage ``Vth`` [V] shared
            across all cells before mismatch is applied.
        vth_mismatch__V: Additive Gaussian mismatch on ``Vth`` [V].
        enable_vth_mismatch: Apply ``vth_mismatch__V`` per cell at
            sampling time.
    """

    # --- Nominal threshold ---
    vth_nominal__V: float

    # --- V_th mismatch ---
    vth_mismatch__V: float
    enable_vth_mismatch: bool

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._require_nonneg(self.vth_mismatch__V, "vth_mismatch__V")


class Selector(nn.Module):
    """OTS selector with training/eval-mode-dependent threshold variation."""

    def __init__(
        self,
        *,
        cfg: SelectorConfig,
        T__K: float,
        dtype: torch.dtype,
        array_shape: tuple[int, ...],
    ) -> None:
        """Initialize the selector model."""
        super().__init__()
        self.cfg = cfg
        self.T__K = T__K
        self.dtype = dtype
        self.register_buffer("vth_nominal__V", torch.full(array_shape, cfg.vth_nominal__V))
        self.register_buffer("_vth_static__V", torch.empty(array_shape), persistent=False)
        self._static_initialized = False

    @property
    def vth_nominal__V_tensor(self) -> Tensor:
        """Typed accessor for nominal Vth buffer."""
        return cast(Tensor, self._buffers["vth_nominal__V"])

    @property
    def vth_static__V_tensor(self) -> Tensor:
        """Typed accessor for static Vth buffer."""
        return cast(Tensor, self._buffers["_vth_static__V"])

    def _sample_vth(self) -> Tensor:
        return apply_gaussian(
            self.vth_nominal__V_tensor,
            self.cfg.vth_mismatch__V,
            enabled=self.cfg.enable_vth_mismatch,
        )

    def sample_vth_like(self, reference: Tensor) -> Tensor:
        """Return a threshold voltage tensor compatible with ``reference``.

        In training mode a new mismatch sample is drawn each call.  In eval
        mode the static map is lazily initialised on the first call then
        reused, then broadcast to ``reference.shape``.

        Args:
            reference: Tensor whose shape, device, and dtype define the
                target threshold tensor (typically the cell-voltage tensor).

        Returns:
            Threshold voltage tensor [V].  Shape: ``reference.shape``.
        """
        if self.training:
            vth = self._sample_vth()
        else:
            if not self._static_initialized:
                self.vth_static__V_tensor.copy_(self._sample_vth())
                self._static_initialized = True
            vth = self.vth_static__V_tensor
        vth = vth.to(device=reference.device, dtype=reference.dtype)
        return torch.broadcast_to(vth, reference.shape)
