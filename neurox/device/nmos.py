"""Continuous EKV-softplus NMOS electrical primitive.

See also:
    docs/dev/modules/device/nmos.md
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurox.common.mixin import FabricateMixin, ValidateMixin
from neurox.common.nonideality import apply_gaussian
from neurox.common.physical_constant import thermal_voltage__V


@dataclass(frozen=True)
class NMOSConfig(ValidateMixin):
    """Immutable PDK config for an NMOS transistor.

    Attributes:
        mu0__cm2_per_V_s: Low-field carrier mobility at ``T_ref__K``
            [cm²/V/s].
        c_ox__fF_per_um2: Gate-oxide capacitance per unit area
            [fF/μm²].
        vth0__V: Nominal threshold voltage at ``T_ref__K`` [V].
        n_factor: SPICE NFACTOR (subthreshold swing coefficient).
            ``> 1.0`` (ideal 60 mV/dec is the 1.0 limit).
        T_ref__K: Reference temperature [K] at which ``mu0`` and
            ``vth0`` are stated.
        ute: Mobility temperature exponent, per
            ``μ(T) = μ0 · (T / T_ref)^(-ute)``.
        kt1__V: V_th temperature coefficient [V], per
            ``V_th(T) = vth0 + kt1 · (T / T_ref - 1)``.
        A_vt__mV_um: Pelgrom V_th matching coefficient [mV·μm];
            ``σ_Vt = A_vt · 1e-3 / sqrt(W · L)``.
        A_beta_relative__um: Pelgrom relative-β matching coefficient
            [μm]; ``σ_β / β = A_beta_relative / sqrt(W · L)``.
        enable_A_vt_mismatch: Apply ``A_vt`` Pelgrom mismatch at
            fabricate time.
        enable_A_beta_mismatch: Apply ``A_beta_relative`` Pelgrom
            mismatch at fabricate time.
    """

    # --- Process electrical ---
    mu0__cm2_per_V_s: float
    c_ox__fF_per_um2: float
    vth0__V: float

    # --- Subthreshold ---
    n_factor: float

    # --- Temperature coefficients ---
    T_ref__K: float
    ute: float
    kt1__V: float

    # --- V_th Pelgrom mismatch ---
    A_vt__mV_um: float
    enable_A_vt_mismatch: bool

    # --- β Pelgrom mismatch ---
    A_beta_relative__um: float
    enable_A_beta_mismatch: bool

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_process()
        self.validate_temperature()
        self.validate_mismatch()

    def validate_process(self) -> None:
        self._require_pos(self.mu0__cm2_per_V_s, "mu0__cm2_per_V_s")
        self._require_pos(self.c_ox__fF_per_um2, "c_ox__fF_per_um2")
        if not (self.n_factor > 1.0):
            raise ValueError(f"require: n_factor ({self.n_factor}) > 1.0")

    def validate_temperature(self) -> None:
        self._require_pos(self.T_ref__K, "T_ref__K")

    def validate_mismatch(self) -> None:
        self._require_nonneg(self.A_vt__mV_um, "A_vt__mV_um")
        self._require_nonneg(self.A_beta_relative__um, "A_beta_relative__um")


@dataclass(frozen=True)
class NMOSDCOP:
    """Solver-facing working-point result for one NMOS evaluation.

    Attributes:
        ids__uA: Drain-source current [uA] — positive drain → source.
        did_dvg__uS: ``∂I_ds/∂V_g`` [uS] = ``gm``.
        did_dvd__uS: ``∂I_ds/∂V_d`` [uS] (non-negative).
        did_dvs__uS: ``∂I_ds/∂V_s`` [uS] (non-positive).
    """

    ids__uA: Tensor
    did_dvg__uS: Tensor
    did_dvd__uS: Tensor
    did_dvs__uS: Tensor


@dataclass(frozen=True)
class NMOSSnapshot:
    """Per-call NMOS state snapshot.

    Attributes:
        beta__uA_per_V2: Per-cell transconductance factor [uA/V²].
        vth__V: Per-cell threshold voltage [V].
    """

    beta__uA_per_V2: Tensor
    vth__V: Tensor


class NMOS(FabricateMixin, nn.Module):
    """EKV-softplus NMOS electrical primitive.

    Args:
        cfg: Immutable PDK NMOS configuration.
        inst_shape: Per-instance fabrication shape.
        dtype: Floating-point dtype for registered buffers.
        T__K: Operating temperature [K].
        W__um: Channel width [μm].
        L__um: Channel length [μm].
    """

    nominal_beta__uA_per_V2: Tensor
    nominal_vth__V: Tensor
    beta__uA_per_V2: Tensor
    vth__V: Tensor

    def __init__(
        self,
        *,
        cfg: NMOSConfig,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        W__um: float,
        L__um: float,
    ) -> None:
        super().__init__()

        if not (W__um > 0.0):
            raise ValueError(f"require: W__um ({W__um}) > 0.0")
        if not (L__um > 0.0):
            raise ValueError(f"require: L__um ({L__um}) > 0.0")

        self.cfg = cfg
        self._inst_shape = inst_shape
        self.W__um = W__um
        self.L__um = L__um
        self.T__K = T__K
        self.dtype = dtype

        T_ratio = T__K / cfg.T_ref__K
        mu_scale = math.pow(T_ratio, -cfg.ute)
        vth_shift__V = cfg.kt1__V * (T_ratio - 1.0)

        # --- Derived electrical constants ---

        # Smoothing scale used by softplus and sigmoid.
        self._inv_smooth_scale__per_V = 1.0 / (2.0 * cfg.n_factor * thermal_voltage__V(T__K))

        # Nominal parameter
        nominal_mu__cm2_per_V_s = cfg.mu0__cm2_per_V_s * mu_scale
        nominal_beta__uA_per_V2 = nominal_mu__cm2_per_V_s * cfg.c_ox__fF_per_um2 * 0.1 * (W__um / L__um)
        nominal_vth__V = cfg.vth0__V + vth_shift__V

        self.register_buffer(
            "nominal_beta__uA_per_V2",
            torch.tensor(nominal_beta__uA_per_V2, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "nominal_vth__V",
            torch.tensor(nominal_vth__V, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "beta__uA_per_V2",
            self.nominal_beta__uA_per_V2.clone(),
            persistent=False,
        )
        self.register_buffer(
            "vth__V",
            self.nominal_vth__V.clone(),
            persistent=False,
        )

        # Pelgrom area-scaled sigmas precomputed once.
        nominal_isqrt_area__per_um = 1.0 / math.sqrt(W__um * L__um)
        self.sigma_vth__V = cfg.A_vt__mV_um * 1e-3 * nominal_isqrt_area__per_um
        self.sigma_beta__uA_per_V2 = nominal_beta__uA_per_V2 * cfg.A_beta_relative__um * nominal_isqrt_area__per_um

    def _sample_fabricate_mismatch(self) -> None:
        """Resample β and V_th at ``self._inst_shape`` (re-callable)."""
        self.beta__uA_per_V2 = apply_gaussian(
            self.nominal_beta__uA_per_V2.clone().expand(self._inst_shape),
            self.sigma_beta__uA_per_V2,
            enabled=self.cfg.enable_A_beta_mismatch,
        )
        self.vth__V = apply_gaussian(
            self.nominal_vth__V.clone().expand(self._inst_shape),
            self.sigma_vth__V,
            enabled=self.cfg.enable_A_vt_mismatch,
        )

    def snapshot(self, *, shape: tuple[int, ...]) -> NMOSSnapshot:
        """Sample one per-call runtime snapshot over ``shape``.

        Args:
            shape: Snapshot shape.

        Returns:
            Per-call snapshot of the fabricated state.
        """
        del shape
        return NMOSSnapshot(beta__uA_per_V2=self.beta__uA_per_V2, vth__V=self.vth__V)

    def solve_dc(
        self,
        vg__V: Tensor | float,
        vd__V: Tensor | float,
        vs__V: Tensor | float,
        snapshot: NMOSSnapshot,
    ) -> NMOSDCOP:
        """Evaluate ``I_ds`` and its three node partials at one op point.

        Args:
            vg__V: Gate voltage [V].
            vd__V: Drain voltage [V].
            vs__V: Source voltage [V].
            snapshot: Per-call NMOS snapshot carrying ``β`` and ``V_th``.

        Returns:
            :class:`NMOSDCOP` with ``ids__uA`` and ``∂I/∂{V_g, V_d, V_s}``.
        """
        beta__uA_per_V2 = snapshot.beta__uA_per_V2
        vth__V = snapshot.vth__V
        inv_smooth_scale__per_V = self._inv_smooth_scale__per_V

        v_ov_s__V = vg__V - vs__V - vth__V
        v_eff_s = F.softplus(v_ov_s__V, beta=inv_smooth_scale__per_V)
        sigma_s = torch.sigmoid(v_ov_s__V * inv_smooth_scale__per_V)

        v_ov_d__V = vg__V - vd__V - vth__V
        v_eff_d = F.softplus(v_ov_d__V, beta=inv_smooth_scale__per_V)
        sigma_d = torch.sigmoid(v_ov_d__V * inv_smooth_scale__per_V)

        ids__uA = 0.5 * beta__uA_per_V2 * (v_eff_s * v_eff_s - v_eff_d * v_eff_d)
        did_dvg__uS = beta__uA_per_V2 * (v_eff_s * sigma_s - v_eff_d * sigma_d)
        did_dvd__uS = beta__uA_per_V2 * v_eff_d * sigma_d
        did_dvs__uS = -beta__uA_per_V2 * v_eff_s * sigma_s

        return NMOSDCOP(
            ids__uA=ids__uA,
            did_dvg__uS=did_dvg__uS,
            did_dvd__uS=did_dvd__uS,
            did_dvs__uS=did_dvs__uS,
        )
