"""OTS (Ovonic Threshold Switch) selector device model for crossbar arrays.

Physical model overview
-----------------------
An OTS selector is a two-terminal chalcogenide device that remains in a
high-resistance state until the applied voltage exceeds a threshold
``Vth`` [V], at which point it switches to a low-resistance conducting state.
In the crossbar context the selector suppresses sneak-path leakage through
half-selected cells.

This module models only the threshold voltage distribution across an array —
the actual switching I-V characteristic is handled by the crossbar solver.

**Threshold voltage mismatch**

    Vth ~ N(vth_nominal, sigma^2)

Mismatch originates from film-thickness and composition variation across the
die.  Two modes are supported, controlled by ``nn.Module.training``:

* *Training mode*: a fresh Gaussian sample is drawn on every call to
  ``sample_vth_like``, exposing the optimiser to the full mismatch
  distribution.
* *Eval mode*: the mismatch is drawn once and frozen in a persistent buffer
  (``_vth_static__V``), modelling the fixed threshold map of a fabricated
  array at inference time.
"""

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from neurox.common.nonideality import apply_gaussian


@dataclass(frozen=True)
class SelectorConfig:
    """Immutable configuration for an OTS threshold selector.

    Attributes:
        vth_nominal__V: Nominal threshold voltage ``Vth`` [V] shared across
            all cells before mismatch is applied.
        vth_mismatch: Additive Gaussian mismatch on ``Vth`` [V]; sigma
            captures within-die process variation.  ``None`` disables
            mismatch — every cell uses the nominal threshold.
    """

    vth_nominal__V: float
    vth_mismatch: float | None = None


class Selector(nn.Module):
    """OTS threshold selector with training/eval-mode-dependent Vth variation.

    In training mode each forward call samples a fresh mismatch realisation.
    In eval mode the mismatch is drawn once on the first call and cached in
    ``_vth_static__V``, representing a fixed fabricated array.

    The nominal threshold is stored as a persistent buffer (saved with
    ``state_dict``); the static mismatch realisation is non-persistent.
    """

    def __init__(self, config: SelectorConfig, array_shape: tuple[int, ...]) -> None:
        """Initialize selector mismatch model.

        Args:
            config: Selector configuration.
            array_shape: Spatial shape for static mismatch realization.
        """
        super().__init__()
        self.config = config
        self.register_buffer("vth_nominal__V", torch.full(array_shape, config.vth_nominal__V))
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
        if self.config.vth_mismatch is None:
            return self.vth_nominal__V_tensor
        return apply_gaussian(self.vth_nominal__V_tensor, self.config.vth_mismatch)

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
