"""Continuous EKV-softplus MOSFET electrical primitive.

See Also:
    docs/reference/primitive/device/mosfet.md
    docs/internals/primitive/device/mosfet.md
"""

import math
from abc import ABC, abstractmethod
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase, TensorDataClassBase, TensorGroupMixin
from neurox.primitive.nonideality import apply_gaussian
from neurox.primitive.physics import thermal_voltage__V


class MosfetConfig(ConfigBase):
    """Immutable PDK config for a MOSFET (polarity-agnostic).

    The same field set describes n- and p-channel devices: `mu0` and `c_ox`
    are positive magnitudes, and `vth0` is a signed threshold whose sign is
    set by the device flavor (enhancement / depletion), not by channel
    polarity.
    """

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
    """Per-source toggles selecting which MOSFET nonidealities are active."""

    A_vt_mismatch: bool
    """Apply Pelgrom V_th mismatch at fabricate time."""
    A_beta_mismatch: bool
    """Apply Pelgrom β mismatch at fabricate time."""


class MosfetDcop(TensorDataClassBase):
    """Caller-facing working-point result for one MOSFET evaluation."""

    ids__uA: Tensor
    """Drain-source current, positive for drain → source flow; a p-channel
    device in normal conduction is typically negative. Shape: `[...]`."""
    did_dvg__uS: Tensor
    """`∂I_ds/∂V_g`, the transconductance `gm`. Shape: `[...]`."""
    did_dvd__uS: Tensor
    """`∂I_ds/∂V_d`, non-negative for both polarities. Shape: `[...]`."""
    did_dvs__uS: Tensor
    """`∂I_ds/∂V_s`, non-positive for both polarities. Shape: `[...]`."""


class MosfetSnap(TensorDataClassBase, TensorGroupMixin):
    """Per-call MOSFET state snap."""

    beta__uA_per_V2: Tensor
    """Per-cell transconductance-factor magnitude, polarity sign excluded.
    Shape: `[...]`."""
    vth__V: Tensor
    """Per-cell signed threshold voltage. Shape: `[...]`."""


class Mosfet(ModuleBase[MosfetConfig, MosfetPolicy], ABC):
    """Polarity-parameterized EKV-softplus MOSFET.

    Args:
        config: PDK parameters and matching coefficients.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication multiplicity.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        W__um: Channel width.
        L__um: Channel length.
    """

    is_profile_target: ClassVar[bool] = False

    # === Nominal buffers ===

    _nominal_beta__uA_per_V2: Tensor  # Shape: []
    _nominal_vth__V: Tensor  # Shape: []

    # === Fabricated state ===

    _beta__uA_per_V2: Tensor  # Shape: [*inst_shape]
    _vth__V: Tensor  # Shape: [*inst_shape]

    def __init__(
        self,
        *,
        config: MosfetConfig,
        policy: MosfetPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        W__um: float,
        L__um: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

        if self.polarity not in (1, -1):
            raise ValueError(f"require: polarity ({self.polarity}) in (1, -1)")
        if not (W__um > 0.0):
            raise ValueError(f"require: W__um ({W__um}) > 0.0")
        if not (L__um > 0.0):
            raise ValueError(f"require: L__um ({L__um}) > 0.0")

        temperature_ratio = T__K / config.T_nom__K
        mu_scale = math.pow(temperature_ratio, -config.ute)
        vth_shift__V = config.kt1__V * (temperature_ratio - 1.0)

        # Smoothing scale used by softplus and sigmoid.
        self._inv_smooth_scale__per_V = 1.0 / (2.0 * config.n_factor * thermal_voltage__V(T__K))

        # Nominal parameter — β is a positive magnitude; the polarity sign
        # is applied in the I-V law, not baked into β.
        nominal_mu__cm2_per_V_s = config.mu0__cm2_per_V_s * mu_scale
        nominal_beta__uA_per_V2 = nominal_mu__cm2_per_V_s * config.c_ox__fF_per_um2 * 0.1 * (W__um / L__um)
        nominal_vth__V = config.vth0__V + vth_shift__V

        self._register_fabrication_buffers(
            dtype=dtype,
            nominal_beta__uA_per_V2=nominal_beta__uA_per_V2,
            nominal_vth__V=nominal_vth__V,
        )

        # Pelgrom area-scaled sigma precomputed once.
        nominal_isqrt_area__per_um = 1.0 / math.sqrt(W__um * L__um)
        self._sigma_vth__V = config.A_vt__mV_um * 1e-3 * nominal_isqrt_area__per_um
        self._sigma_beta__uA_per_V2 = nominal_beta__uA_per_V2 * config.A_beta_relative__um * nominal_isqrt_area__per_um

    @property
    @abstractmethod
    def polarity(self) -> int:
        """Channel polarity sign: `+1` (n-channel) or `-1` (p-channel)."""
        raise NotImplementedError

    def _register_fabrication_buffers(
        self,
        *,
        dtype: torch.dtype,
        nominal_beta__uA_per_V2: float,
        nominal_vth__V: float,
    ) -> None:
        """Register immutable tensors used as fabrication sources."""
        self.register_buffer(
            "_nominal_beta__uA_per_V2",
            torch.tensor(nominal_beta__uA_per_V2, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "_nominal_vth__V",
            torch.tensor(nominal_vth__V, dtype=dtype),
            persistent=False,
        )

    def _sample_fabricate_mismatch(self) -> None:
        self._beta__uA_per_V2 = apply_gaussian(
            self._nominal_beta__uA_per_V2.clone().expand(self.inst_shape),
            self._sigma_beta__uA_per_V2,
            enabled=self.policy.A_beta_mismatch,
        )
        self._vth__V = apply_gaussian(
            self._nominal_vth__V.clone().expand(self.inst_shape),
            self._sigma_vth__V,
            enabled=self.policy.A_vt_mismatch,
        )

    def snapshot(
        self,
        *,
        shape: tuple[int, ...],
    ) -> MosfetSnap:
        """Sample one per-call runtime snap of the fabricated state.

        Args:
            shape: Broadcast shape the snap's tensor fields are filled at.

        Returns:
            Per-call β and V_th views.
        """
        vth_view = self._vth__V.expand(shape) if shape else self._vth__V
        beta_view = self._beta__uA_per_V2.expand(shape) if shape else self._beta__uA_per_V2
        return MosfetSnap(vth__V=vth_view, beta__uA_per_V2=beta_view)

    def solve_dc(
        self,
        vg__V: Tensor | float,
        vd__V: Tensor | float,
        vs__V: Tensor | float,
        snap: MosfetSnap,
    ) -> MosfetDcop:
        """Evaluate `I_ds` and its three node partials at one op point.

        Args:
            vg__V: Gate voltage.
            vd__V: Drain voltage.
            vs__V: Source voltage.
            snap: Per-call MOSFET snap carrying β and V_th.

        Returns:
            Drain-source current and its three node partials.
        """
        p = self.polarity
        beta__uA_per_V2 = snap.beta__uA_per_V2
        vth__V = snap.vth__V
        inv_smooth_scale__per_V = self._inv_smooth_scale__per_V

        # --- 1: evaluate source-side smoothed voltage ---

        v_ov_s__V = p * (vg__V - vs__V - vth__V)
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

        return MosfetDcop(
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
