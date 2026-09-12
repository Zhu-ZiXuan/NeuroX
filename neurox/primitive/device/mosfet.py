"""Continuous EKV-softplus MOSFET electrical primitive.

See Also:
    docs/reference/primitive/device/mosfet.md
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.common.module import ConfigBase, DcopBase, ModuleBase, PolicyBase, SnapBase
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


class Mosfet(ModuleBase, ABC):
    """Polarity-parameterized EKV-softplus MOSFET.

    Args:
        W__um: Channel width.
        L__um: Channel length.
    """

    is_profile_target: ClassVar[bool] = False

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
        dtype: torch.dtype,
        T__K: float,
        W__um: float,
        L__um: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

        if self.polarity not in (1, -1):
            raise ValueError(f"require: polarity ({self.polarity}) in (1, -1)")
        if not (math.isfinite(T__K) and T__K > 0.0):
            raise ValueError(f"require: T__K ({T__K}) finite and > 0.0")
        if not (math.isfinite(W__um) and W__um > 0.0):
            raise ValueError(f"require: W__um ({W__um}) > 0.0")
        if not (math.isfinite(L__um) and L__um > 0.0):
            raise ValueError(f"require: L__um ({L__um}) > 0.0")
        if not dtype.is_floating_point:
            raise TypeError(f"Mosfet requires a floating-point dtype; got {dtype}")
        dtype_info = torch.finfo(dtype)

        temperature_ratio = T__K / config.T_nom__K
        mu_scale = math.pow(temperature_ratio, -config.ute)
        vth_shift__V = config.kt1__V * (temperature_ratio - 1.0)

        # Smoothing scale used by softplus and sigmoid.
        self._inv_smooth_scale__per_V = 1.0 / (2.0 * config.n_factor * thermal_voltage__V(T__K))

        # β stays a positive magnitude; the polarity sign is applied in the
        # I-V law, not baked into β.
        nominal_mu__cm2_per_V_s = config.mu0__cm2_per_V_s * mu_scale
        # 0.1 reconciles the mixed unit systems of the product: mobility is in
        # cm^2 while c_ox and W/L are per um^2, and β must come out in uA/V^2 —
        # 1e8 (cm^2 -> um^2) · 1e-15 (fF -> F) · 1e6 (A -> uA) = 0.1.
        nominal_beta__uA_per_V2 = nominal_mu__cm2_per_V_s * config.c_ox__fF_per_um2 * 0.1 * (W__um / L__um)
        nominal_vth__V = config.vth0__V + vth_shift__V
        if not (dtype_info.tiny <= nominal_beta__uA_per_V2 <= dtype_info.max):
            raise ValueError(
                f"nominal_beta__uA_per_V2 ({nominal_beta__uA_per_V2}) is not a positive normal value "
                f"representable by {dtype}"
            )

        self._register_fabrication_buffers(
            dtype=dtype,
            nominal_beta__uA_per_V2=nominal_beta__uA_per_V2,
            nominal_vth__V=nominal_vth__V,
        )

        # Pelgrom area-scaled sigma precomputed once.
        nominal_isqrt_area__per_um = 1.0 / math.sqrt(W__um * L__um)
        self._sigma_vth__V = config.A_vt__mV_um * 1e-3 * nominal_isqrt_area__per_um
        self._sigma_beta__uA_per_V2 = nominal_beta__uA_per_V2 * config.A_beta_relative__um * nominal_isqrt_area__per_um
        if not (math.isfinite(self._sigma_beta__uA_per_V2) and self._sigma_beta__uA_per_V2 <= dtype_info.max):
            raise ValueError(f"sigma_beta__uA_per_V2 ({self._sigma_beta__uA_per_V2}) is not representable by {dtype}")

    @property
    @abstractmethod
    def polarity(self) -> int:
        """Channel polarity sign: `+1` (n-channel) or `-1` (p-channel).

        The concrete class fixes it, so config, policy, snap, and result stay
        polarity-free and one model core serves both channel types.
        """
        raise NotImplementedError

    def _register_fabrication_buffers(
        self,
        *,
        dtype: torch.dtype,
        nominal_beta__uA_per_V2: float,
        nominal_vth__V: float,
    ) -> None:
        self._register_nonpersistent_buffer(
            "_nominal_beta__uA_per_V2",
            torch.tensor(nominal_beta__uA_per_V2, dtype=dtype),
        )
        self._register_nonpersistent_buffer(
            "_nominal_vth__V",
            torch.tensor(nominal_vth__V, dtype=dtype),
        )

    def _sample_fabrication_variation(self) -> None:
        beta__uA_per_V2 = apply_gaussian(
            self._nominal_beta__uA_per_V2.clone().expand(self.inst_shape),
            self._sigma_beta__uA_per_V2,
            enabled=self.policy.A_beta_mismatch,
        )
        if self.policy.A_beta_mismatch:
            dtype_info = torch.finfo(beta__uA_per_V2.dtype)
            beta__uA_per_V2 = beta__uA_per_V2.clamp(min=dtype_info.tiny, max=dtype_info.max)
        self._beta__uA_per_V2 = beta__uA_per_V2
        self._vth__V = apply_gaussian(
            self._nominal_vth__V.clone().expand(self.inst_shape),
            self._sigma_vth__V,
            enabled=self.policy.A_vt_mismatch,
        )

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
        vg__V: Tensor | float,
        vd__V: Tensor | float,
        vs__V: Tensor | float,
        snap: _Snap,
    ) -> _Dcop:
        """Evaluate `I_ds` and its three node partials at one op point."""
        p = self.polarity
        beta__uA_per_V2 = snap.beta__uA_per_V2
        vth__V = snap.vth__V
        inv_smooth_scale__per_V = self._inv_smooth_scale__per_V

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


class Nmos(Mosfet):
    """N-channel MOSFET."""

    polarity = 1


class Pmos(Mosfet):
    """P-channel MOSFET."""

    polarity = -1
