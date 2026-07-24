"""Continuous EKV-softplus MOSFET electrical primitive.

See also:
    docs/reference/primitive/device/mosfet.md
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor

from neurox.common import ConfigBase, ModuleBase, PolicyBase
from neurox.primitive.nonideality import apply_gaussian
from neurox.primitive.physical_constant import thermal_voltage__V


class MosfetConfig(ConfigBase):
    """Immutable PDK config for a MOSFET (polarity-agnostic).

    The same field set describes n- and p-channel devices: ``mu0`` and
    ``c_ox`` are positive magnitudes, and ``vth0`` is a signed threshold
    whose sign is set by the device flavor (enhancement / depletion), not by
    channel polarity.

    Attributes:
        T_nom__K: Reference temperature at which ``mu0`` and ``vth0`` are stated.
        c_ox__fF_per_um2: Gate-oxide capacitance area density.
        mu0__cm2_per_V_s: Low-field carrier mobility.
        ute: Mobility temperature exponent.
        vth0__V: Threshold voltage.
        kt1__V: Threshold voltage temperature coefficient.
        n_factor: Subthreshold swing coefficient.
        A_vt__mV_um: Pelgrom V_th matching coefficient.
        A_beta_relative__um: Pelgrom relative-β matching coefficient.
    """

    T_nom__K: float
    c_ox__fF_per_um2: float

    mu0__cm2_per_V_s: float
    ute: float

    vth0__V: float
    kt1__V: float

    n_factor: float

    A_vt__mV_um: float
    A_beta_relative__um: float

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
    """Per-source toggles selecting which MOSFET nonidealities are active.

    Attributes:
        A_vt_mismatch: Apply ``A_vt`` Pelgrom V_th mismatch at fabricate time.
        A_beta_mismatch: Apply ``A_beta_relative`` Pelgrom β mismatch at fabricate time.
    """

    A_vt_mismatch: bool
    A_beta_mismatch: bool


@dataclass(frozen=True)
class MosfetDcop:
    """Caller-facing working-point result for one MOSFET evaluation.

    Attributes:
        ids__uA: Drain-source current — positive for drain → source
            flow. For a p-channel device in normal conduction ``ids__uA``
            is typically negative (real flow is source → drain).
        did_dvg__uS: ``∂I_ds/∂V_g`` = ``gm``.
        did_dvd__uS: ``∂I_ds/∂V_d`` (non-negative for both polarities).
        did_dvs__uS: ``∂I_ds/∂V_s`` (non-positive for both polarities).
    """

    ids__uA: Tensor
    did_dvg__uS: Tensor
    did_dvd__uS: Tensor
    did_dvs__uS: Tensor


@dataclass(frozen=True)
class MosfetSnap:
    """Per-call MOSFET state snap.

    Attributes:
        beta__uA_per_V2: Per-cell transconductance-factor magnitude.
        vth__V: Per-cell signed threshold voltage.
    """

    beta__uA_per_V2: Tensor
    vth__V: Tensor


class Mosfet(ModuleBase[MosfetConfig, MosfetPolicy], ABC):
    """Polarity-parameterized EKV-softplus MOSFET.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
        W__um: Channel width.
        L__um: Channel length.
    """

    is_profile_target: ClassVar[bool] = False

    # --- Fabrication source buffers ---

    _nominal_beta__uA_per_V2: Tensor
    _nominal_vth__V: Tensor

    @property
    @abstractmethod
    def polarity(self) -> int:
        """Channel polarity sign: ``+1`` (n-channel) or ``-1`` (p-channel)."""

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

        T_ratio = T__K / config.T_nom__K
        mu_scale = math.pow(T_ratio, -config.ute)
        vth_shift__V = config.kt1__V * (T_ratio - 1.0)

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
        multi_coords: tuple[Tensor, ...] | None,
    ) -> MosfetSnap:
        """Sample one per-call runtime snap over ``shape``.

        Args:
            shape: Per-call broadcast shape; the snap fills tensor
                fields at this shape.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view; ``None`` returns the
                full view.

        Returns:
            Per-call snap of the fabricated state.
        """
        vth_view = self._vth__V.expand(shape) if shape else self._vth__V
        beta_view = self._beta__uA_per_V2.expand(shape) if shape else self._beta__uA_per_V2
        if multi_coords is None:
            return MosfetSnap(vth__V=vth_view, beta__uA_per_V2=beta_view)
        return MosfetSnap(
            vth__V=vth_view[multi_coords],
            beta__uA_per_V2=beta_view[multi_coords],
        )

    def solve_dc(
        self,
        vg__V: Tensor | float,
        vd__V: Tensor | float,
        vs__V: Tensor | float,
        snap: MosfetSnap,
    ) -> MosfetDcop:
        """Evaluate ``I_ds`` and its three node partials at one op point.

        Args:
            vg__V: Gate voltage.
            vd__V: Drain voltage.
            vs__V: Source voltage.
            snap: Per-call MOSFET snap carrying ``β`` and ``V_th``.

        Returns:
            :class:`MosfetDcop`.
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
