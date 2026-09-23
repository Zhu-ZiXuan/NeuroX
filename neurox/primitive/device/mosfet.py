"""Continuous EKV-softplus MOSFET electrical primitive.

See Also:
    docs/reference/primitive/device/mosfet.md
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.common.module import ConfigBase, DcopBase, NonProfileModule, PolicyBase, SnapBase
from neurox.primitive.nonideality import apply_gaussian
from neurox.primitive.physics import thermal_voltage__V

__all__ = [
    "Mosfet",
    "MosfetConfig",
    "MosfetDcop",
    "MosfetPolicy",
    "MosfetSnap",
    "Nmos",
    "Pmos",
]


class MosfetConfig(ConfigBase):
    # === Process ===

    T_nom__K: float
    """Reference temperature at which `mu0__cm2_per_V_s` and `vth0__V` are stated."""
    c_ox__fF_per_um2: float
    """Gate-oxide capacitance per unit gate area."""
    mu0__cm2_per_V_s: float
    """Low-field carrier mobility."""
    ute: float
    """Mobility temperature exponent."""
    vth0__V: float
    """Threshold voltage at `T_nom__K`."""
    kt1__V: float
    """Threshold voltage temperature coefficient."""
    n_factor: float
    """Subthreshold swing coefficient."""

    # === Mismatch ===

    A_vt__mV_um: float
    """Pelgrom V_th matching coefficient."""
    A_beta_relative__um: float
    """Pelgrom relative-β matching coefficient."""

    def validate(self) -> None:

        # --- Process ---

        self._require_pos(self.T_nom__K, "T_nom__K")
        self._require_pos(self.c_ox__fF_per_um2, "c_ox__fF_per_um2")
        self._require_pos(self.mu0__cm2_per_V_s, "mu0__cm2_per_V_s")
        self._require_gt(self.n_factor, "n_factor", 1.0)

        # --- Mismatch ---

        self._require_non_neg(self.A_vt__mV_um, "A_vt__mV_um")
        self._require_non_neg(self.A_beta_relative__um, "A_beta_relative__um")


class MosfetPolicy(PolicyBase):
    A_vt_mismatch: bool
    """Apply Pelgrom V_th mismatch at fabricate time."""
    A_beta_mismatch: bool
    """Apply Pelgrom β mismatch at fabricate time."""


class MosfetDcop(DcopBase):
    ids__uA: Tensor
    """Drain-source current, positive for drain → source flow; a p-channel
    device in normal conduction is typically negative."""
    did_dvg__uS: Tensor
    """`∂I_ds/∂V_g`, the transconductance `gm`."""
    did_dvd__uS: Tensor
    """`∂I_ds/∂V_d`, non-negative for both polarities."""
    did_dvs__uS: Tensor
    """`∂I_ds/∂V_s`, non-positive for both polarities."""


class MosfetSnap(SnapBase):
    beta__uA_per_V2: Tensor
    """Per-cell transconductance-factor magnitude, polarity sign excluded."""
    vth__V: Tensor
    """Per-cell signed threshold voltage."""


_Config = MosfetConfig
_Policy = MosfetPolicy
_Dcop = MosfetDcop
_Snap = MosfetSnap


class Mosfet(NonProfileModule, ABC, base_only=True):
    """Polarity-parameterized EKV-softplus MOSFET.

    Args:
        W__um: Channel width.
        L__um: Channel length.
    """

    config: _Config
    policy: _Policy

    # === Nominal buffers ===

    _nominal_beta__uA_per_V2: Tensor  # Shape: []
    _nominal_vth__V: Tensor  # Shape: []

    # === Fabricated state ===

    _beta__uA_per_V2: Tensor  # Shape: [*inst_shape]
    _vth__V: Tensor  # Shape: [*inst_shape]

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        W__um: float,
        L__um: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

        if self.polarity not in (1, -1):
            raise ValueError(f"require: polarity ({self.polarity}) in (1, -1)")
        if not (math.isfinite(W__um) and W__um > 0.0):
            raise ValueError(f"require: W__um ({W__um}) > 0.0")
        if not (math.isfinite(L__um) and L__um > 0.0):
            raise ValueError(f"require: L__um ({L__um}) > 0.0")
        if not dtype.is_floating_point:
            raise TypeError(f"Mosfet requires a floating-point dtype; got {dtype}")

        self._w_l_ratio = W__um / L__um
        self._isqrt_area__per_um = 1.0 / math.sqrt(W__um * L__um)
        self._sigma_vth__V = config.A_vt__mV_um * 1e-3 * self._isqrt_area__per_um
        self._register_fabrication_buffers(dtype=dtype)

    # === Public API ===

    @torch.no_grad()
    def snapshot(
        self,
        *,
        shape: tuple[int, ...],
    ) -> _Snap:
        """Sample one per-call runtime snap of the fabricated state.

        Args:
            shape: Broadcast shape the snap's tensor fields are filled at.

        Returns:
            Per-call β and V_th views.
        """
        vth_view = self._vth__V.expand(shape) if shape else self._vth__V
        beta_view = self._beta__uA_per_V2.expand(shape) if shape else self._beta__uA_per_V2
        return _Snap(vth__V=vth_view, beta__uA_per_V2=beta_view)

    @torch.no_grad()
    def solve_dc(
        self,
        *,
        vg__V: Tensor | float,
        vd__V: Tensor | float,
        vs__V: Tensor | float,
        snap: _Snap,
    ) -> _Dcop:
        """Evaluate `I_ds` and its three node partials at one op point."""
        config = self.config

        p = self.polarity
        beta__uA_per_V2 = snap.beta__uA_per_V2
        vth__V = snap.vth__V
        inv_smooth_scale__per_V = 1.0 / (2.0 * config.n_factor * thermal_voltage__V(self.T__K))

        # --- 1: evaluate source-side smoothed voltage ---

        v_ov_s__V = p * (vg__V - vs__V - vth__V)
        # `beta=` is softplus's own sharpness keyword, unrelated to the device β.
        v_eff_s__V = F.softplus(v_ov_s__V, beta=inv_smooth_scale__per_V)
        sigma_s = F.sigmoid(v_ov_s__V * inv_smooth_scale__per_V)

        # --- 2: evaluate drain-side smoothed voltage ---

        v_ov_d__V = p * (vg__V - vd__V - vth__V)
        v_eff_d__V = F.softplus(v_ov_d__V, beta=inv_smooth_scale__per_V)
        sigma_d = F.sigmoid(v_ov_d__V * inv_smooth_scale__per_V)

        # --- 3: compute drain-source current ---

        ids__uA = 0.5 * p * beta__uA_per_V2 * (v_eff_s__V * v_eff_s__V - v_eff_d__V * v_eff_d__V)

        # --- 4: compute node derivatives ---

        v_s_sigma_s = v_eff_s__V * sigma_s
        v_d_sigma_d = v_eff_d__V * sigma_d

        did_dvg__uS = beta__uA_per_V2 * (v_s_sigma_s - v_d_sigma_d)
        did_dvd__uS = beta__uA_per_V2 * v_d_sigma_d
        did_dvs__uS = -beta__uA_per_V2 * v_s_sigma_s

        return _Dcop(
            ids__uA=ids__uA,
            did_dvg__uS=did_dvg__uS,
            did_dvd__uS=did_dvd__uS,
            did_dvs__uS=did_dvs__uS,
        )

    # === Required by base class ===

    def _sample_fabrication_variation(self) -> None:
        config = self.config
        policy = self.policy

        # --- 1: apply temperature scaling to the compact reference values ---

        temperature_ratio = self.T__K / config.T_nom__K
        beta__uA_per_V2 = self._nominal_beta__uA_per_V2 * math.pow(temperature_ratio, -config.ute)
        vth__V = self._nominal_vth__V + config.kt1__V * (temperature_ratio - 1.0)
        sigma_beta__uA_per_V2 = beta__uA_per_V2 * config.A_beta_relative__um * self._isqrt_area__per_um

        # --- 2: sample and retain one per-instance realization ---

        beta__uA_per_V2 = apply_gaussian(
            beta__uA_per_V2.expand(self.inst_shape),
            sigma=sigma_beta__uA_per_V2,
            enabled=policy.A_beta_mismatch,
        )
        if policy.A_beta_mismatch:
            dtype_info = torch.finfo(beta__uA_per_V2.dtype)
            beta__uA_per_V2 = beta__uA_per_V2.clamp(min=dtype_info.tiny, max=dtype_info.max)

        self._beta__uA_per_V2 = beta__uA_per_V2
        self._vth__V = apply_gaussian(
            vth__V.expand(self.inst_shape), sigma=self._sigma_vth__V, enabled=policy.A_vt_mismatch
        )

    # === For subclass to implement or override ===

    @property
    @abstractmethod
    def polarity(self) -> int:
        """Channel polarity sign: `+1` (n-channel) or `-1` (p-channel).

        The concrete class fixes it, so config, policy, snap, and result stay
        polarity-free and one model core serves both channel types.
        """
        raise NotImplementedError

    # === Tools for subclass and internal use ===

    def _register_fabrication_buffers(self, *, dtype: torch.dtype) -> None:
        config = self.config
        # Reference values are fixed at config.T_nom__K.
        # The 0.1 factor converts cm^2, fF, and um^2 to uA/V^2.
        nominal_beta__uA_per_V2 = config.mu0__cm2_per_V_s * config.c_ox__fF_per_um2 * 0.1 * self._w_l_ratio
        dtype_info = torch.finfo(dtype)
        if not (dtype_info.tiny <= nominal_beta__uA_per_V2 <= dtype_info.max):
            raise ValueError(
                f"nominal_beta__uA_per_V2 ({nominal_beta__uA_per_V2}) is not a positive normal value "
                f"representable by {dtype}"
            )
        self._register_nonpersistent_buffer(
            "_nominal_beta__uA_per_V2", torch.tensor(nominal_beta__uA_per_V2, dtype=dtype)
        )
        self._register_nonpersistent_buffer("_nominal_vth__V", torch.tensor(config.vth0__V, dtype=dtype))


class Nmos(Mosfet):
    """N-channel MOSFET."""

    polarity = 1


class Pmos(Mosfet):
    """P-channel MOSFET."""

    polarity = -1
