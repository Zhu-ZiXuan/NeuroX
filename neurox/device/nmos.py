"""General-purpose NMOS transistor model with continuous EKV-softplus I-V.

Physical model overview
-----------------------
A single :class:`NMOS` instance models a fabricated transistor (or a per-cell
array of them).  Its purpose is twofold:

1. **General electrical primitive** — :meth:`ids__mA`, :meth:`gds__mS`, and
   :meth:`gm__mS` evaluate the drain-source current and its analytical
   derivatives at any operating point ``(V_g, V_d, V_s)``.  No assumption is
   made about which terminal is the source vs. drain or which region of
   operation the device sits in.
2. **Per-cell device for a 1T1R access transistor** — :meth:`fabricate`
   samples per-cell mismatch (W, L, μ·Cox, V_th) and **returns** the two
   tensors that drive every electrical equation (``β`` and ``V_th``) at
   the requested shape.  The calling :class:`Xbar` subclass registers
   them as its own buffers; downstream consumers (xbar forward, 1T1R
   solver) pass these tensors back into ``ids__uA`` / ``gds__uS`` /
   ``gm__uS`` along with the operating-point voltages.

Continuous EKV-softplus I-V
---------------------------
We use the EKV symmetric formulation with a softplus smoothing of the
overdrive voltage.  Letting ``η = n_factor · V_T`` and
``smooth_scale = 2 · η``,

    V_eff,s = softplus(V_g - V_s - V_th0,  β = 1 / smooth_scale)
    V_eff,d = softplus(V_g - V_d - V_th0,  β = 1 / smooth_scale)
    I_ds    = 0.5 · β · (V_eff,s² - V_eff,d²)

Limits:

* Strong inversion (``V_ov >> 0``): ``softplus(x) → x``, recovering the
  classical Shichman-Hodges square law in linear and saturation regimes.
* Subthreshold (``V_ov << 0``): ``softplus(x) → exp(x)``, recovering the
  diffusion-dominated exponential leakage.

Both regions are captured by one analytic expression that is continuous and
infinitely differentiable everywhere — important for Newton-Raphson solvers
in non-linear circuit nets that otherwise stall at region boundaries.

Analytical derivatives
----------------------
Because ``d(softplus(x)) / dx = sigmoid(x)``, every conductance is closed
form:

    σ_s = sigmoid((V_g - V_s - V_th0) / smooth_scale)
    σ_d = sigmoid((V_g - V_d - V_th0) / smooth_scale)

    g_ds = ∂I_ds / ∂V_ds  =  ½ · β · (V_eff,d · σ_d  +  V_eff,s · σ_s)
    g_m  = ∂I_ds / ∂V_g   =       β · (V_eff,s · σ_s  -  V_eff,d · σ_d)

``g_ds`` is the symmetric Vcm-fixed drain-source small-signal conductance —
the same quantity an MNA solver would stamp between drain and source nodes.
In deep triode (``V_ds ≈ 0``, ``V_ov >> 0``) it reduces to
``β · (V_gs - V_th)`` — the classical "g_on".  In strong subthreshold
(``V_g << V_th``) it reduces to a value proportional to
``β · η · exp((V_g - V_s - V_th0) / η)`` — the diffusion conductance.

Fabrication
-----------
The module keeps a strict **nominal vs. fabricated** split:

* :attr:`nominal_beta__uA_per_V2` and :attr:`nominal_vth__V` are 0-d
  buffers built once at ``__init__`` from the cfg + temperature.
  They are never overwritten.
* :attr:`beta__uA_per_V2` and :attr:`vth__V` are the fabricated
  per-cell buffers.  :meth:`fabricate(shape)` always rebuilds them
  from the nominal templates via ``nominal.clone().expand(shape)``,
  then applies the optional Pelgrom-style Gaussian mismatch.  When
  the corresponding sigma is ``None`` the fabricated buffer remains
  a single-element view of the cloned nominal (no full-shape memory
  is allocated).

Re-calling :meth:`fabricate` always starts from the nominal
templates — the fabricated buffers never feed back into a later
fabrication.  Higher layers never see the buffers directly; they
go through :meth:`snapshot` / :meth:`solve_dc`.

Parasitic capacitances
----------------------
Four lumped caps (``c_gs__fF``, ``c_gd__fF``, ``c_db__fF``, ``c_sb__fF``) are
compiled from geometry + PDK constants in :meth:`__init__` and exposed as
Python-float scalars.  The ``½·W·L·C_ox`` channel split is the exact
deep-triode partitioning; junction caps split into bottom-area and
sidewall-perimeter components.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neurox.common.nonideality import apply_gaussian
from neurox.common.physical_constant import thermal_voltage__V


@dataclass(frozen=True)
class NMOSConfig:
    """Immutable physical PDK config for an NMOS transistor.

    All geometric quantities are in μm and capacitance densities are
    referenced to μm.  Threshold voltage and mobility are stated at
    ``T_ref__K``; the NMOS class scales both to the actual operating
    temperature using BSIM-style coefficients.

    Attributes:
        W__um: Nominal channel width [μm].
        L__um: Nominal channel length [μm].
        A_D__um2: Drain active-area footprint (bottom area) [μm²].
        P_D__um: Drain active-area sidewall perimeter [μm].
        A_S__um2: Source active-area footprint [μm²].
        P_S__um: Source active-area sidewall perimeter [μm].
        mu0__cm2_per_V_s: Low-field carrier mobility at ``T_ref__K``
            [cm²/V/s].  Combined with ``c_ox`` and ``W/L`` gives β.
        c_ox__fF_per_um2: Gate-oxide capacitance per unit area [fF/μm²].
        vth0__V: Nominal threshold voltage at ``T_ref__K`` [V].
        c_gdo__fF_per_um: Gate-drain overlap capacitance per unit
            channel-width [fF/μm].
        c_gso__fF_per_um: Gate-source overlap capacitance per unit
            channel-width [fF/μm].
        c_j__fF_per_um2: Junction bottom-wall capacitance per unit
            area [fF/μm²].
        c_jsw__fF_per_um: Junction sidewall capacitance per unit
            perimeter [fF/μm].
        n_factor: SPICE NFACTOR (subthreshold swing coefficient).
            Must satisfy ``n_factor > 1.0`` (ideal 60 mV/dec is the
            1.0 limit); typical 28 nm RVT value is ≈ 1.25.
        T_ref__K: Reference temperature [K] at which ``mu0`` and
            ``vth0`` are stated.  The init-time ``T__K`` argument is
            scaled against this.
        ute: Mobility temperature exponent (dimensionless), per
            ``μ(T) = μ0 · (T / T_ref)^(-ute)``.  Typical ≈ 1.5.
        kt1__V: V_th temperature coefficient [V], per
            ``V_th(T) = vth0 + kt1 · (T / T_ref - 1)``.  Typically
            ≈ -0.05 V (V_th drops as T rises).
        A_vt__V_um: Pelgrom V_th matching coefficient [V·μm].
            Per-cell ``σ_Vt = A_vt / sqrt(W · L)``.  ``None`` skips
            V_th mismatch entirely.
        A_beta_relative__um: Pelgrom relative-β matching coefficient
            [μm].  Per-cell ``σ_β / β = A_beta_relative / sqrt(W · L)``.
            ``None`` skips β mismatch.
    """

    # --- Geometry ---
    W__um: float
    L__um: float
    A_D__um2: float
    P_D__um: float
    A_S__um2: float
    P_S__um: float

    # --- Process parameters ---
    mu0__cm2_per_V_s: float
    c_ox__fF_per_um2: float
    vth0__V: float

    c_gdo__fF_per_um: float
    c_gso__fF_per_um: float
    c_j__fF_per_um2: float
    c_jsw__fF_per_um: float

    # --- Subthreshold parameters ---
    n_factor: float

    # --- Temperature coefficients ---
    T_ref__K: float
    ute: float
    kt1__V: float

    # --- Fabricate mismatch ---
    A_vt__mV_um: float | None = None
    A_beta_relative__um: float | None = None

    def validate(self) -> None:
        """Validate NMOS configuration parameters."""
        if not (self.W__um > 0.0):
            raise ValueError(f"require: W__um ({self.W__um}) > 0.0")
        if not (self.L__um > 0.0):
            raise ValueError(f"require: L__um ({self.L__um}) > 0.0")
        if not (self.A_D__um2 >= 0.0 and self.P_D__um >= 0.0):
            raise ValueError(f"require: A_D, P_D >= 0, got {self.A_D__um2}, {self.P_D__um}")
        if not (self.A_S__um2 >= 0.0 and self.P_S__um >= 0.0):
            raise ValueError(f"require: A_S, P_S >= 0, got {self.A_S__um2}, {self.P_S__um}")
        if not (self.mu0__cm2_per_V_s > 0.0):
            raise ValueError(f"require: mu0__cm2_per_V_s ({self.mu0__cm2_per_V_s}) > 0.0")
        if not (self.c_ox__fF_per_um2 > 0.0):
            raise ValueError(f"require: c_ox__fF_per_um2 ({self.c_ox__fF_per_um2}) > 0.0")
        if not (self.n_factor > 1.0):
            raise ValueError(f"require: n_factor ({self.n_factor}) > 1.0 (ideal SS=60mV/dec is the limit)")
        if not (self.T_ref__K > 0.0):
            raise ValueError(f"require: T_ref__K ({self.T_ref__K}) > 0.0")
        for name, value in (
            ("c_gdo__fF_per_um", self.c_gdo__fF_per_um),
            ("c_gso__fF_per_um", self.c_gso__fF_per_um),
            ("c_j__fF_per_um2", self.c_j__fF_per_um2),
            ("c_jsw__fF_per_um", self.c_jsw__fF_per_um),
        ):
            if not (value >= 0.0):
                raise ValueError(f"require: {name} ({value}) >= 0.0")
        for name, mis in (
            ("A_vt__mV_um", self.A_vt__mV_um),
            ("A_beta_relative__um", self.A_beta_relative__um),
        ):
            if mis is not None and not (mis >= 0.0):
                raise ValueError(f"require: {name} ({mis}) >= 0.0")

    def __post_init__(self) -> None:
        self.validate()


@dataclass(frozen=True)
class NMOSDC:
    """Solver-facing working-point result for one NMOS evaluation.

    Returned by :meth:`NMOS.solve_dc`; frozen so ``@torch.compile``'s
    Dynamo tracer treats it as an immutable value rather than an
    object whose attributes might mutate.

    All four fields are evaluated at the same operating point
    ``(V_g, V_d, V_s)`` and share the underlying ``V_eff`` / σ
    intermediates that ``solve_dc`` computes once.

    Attributes:
        ids__uA: Drain-source current [uA] — positive when current
            flows drain → source.
        did_dvg__uS: Gate partial ``∂I_ds/∂V_g`` [uS] = ``gm``.
        did_dvd__uS: Drain partial ``∂I_ds/∂V_d`` [uS] (non-negative).
        did_dvs__uS: Source partial ``∂I_ds/∂V_s`` [uS] (non-positive).
    """

    ids__uA: Tensor
    did_dvg__uS: Tensor
    did_dvd__uS: Tensor
    did_dvs__uS: Tensor


@dataclass(frozen=True)
class NMOSSnapshot:
    """Per-VMM NMOS state snapshot.

    Carries the per-VMM materialized device parameters consumed by
    :meth:`NMOS.solve_dc`.  Today this is a thin view over the
    fabricated buffers; future dynamic noise (drain shot, flicker,
    etc.) plugs in by perturbing these fields inside
    :meth:`NMOS.snapshot` before wrapping — no API change needed.

    Attributes:
        beta__uA_per_V2: Per-cell transconductance factor [uA/V²].
        vth__V: Per-cell threshold voltage [V].
    """

    beta__uA_per_V2: Tensor
    vth__V: Tensor


class NMOS(nn.Module):
    """NMOS model (EKV style) — per-instance fabricated state.

    Each :class:`NMOS` instance internally owns its fabricated
    mismatch tensors.  Four non-persistent buffers live on the
    module:

    * ``nominal_beta__uA_per_V2`` — 0-d scalar at the design β.
      Built once at ``__init__`` from cfg + temperature; never
      overwritten.
    * ``nominal_vth__V`` — 0-d scalar at the design V_th.  Same
      lifecycle as ``nominal_beta__uA_per_V2``.
    * ``beta__uA_per_V2`` — fabricated per-cell β [uA/V²].
    * ``vth__V`` — fabricated per-cell V_th [V].

    :meth:`fabricate(shape)` rebuilds the two fabricated buffers
    from the nominal templates via ``nominal.clone().expand(shape)``
    on every call.  Pelgrom-style Gaussian mismatch
    (``σ_Vt = A_vt / sqrt(W · L)``,
    ``σ_β / β = A_beta_relative / sqrt(W · L)``) is sampled once per
    call.  When the matching coefficient is ``None`` the fabricated
    buffer stays a single-element view of the cloned nominal (no
    full-shape memory is allocated).

    Args:
        cfg: Immutable physical NMOS configuration.
        T__K: Operating temperature in Kelvin.  Drives the thermal
            voltage ``V_T = k_B · T / q`` and applies BSIM-style
            temperature scaling to ``μ`` and ``V_th`` against
            ``cfg.T_ref__K``.  Must be ``> 0``.
        dtype: Floating-point dtype used by the registered buffers.
    """

    nominal_beta__uA_per_V2: Tensor
    nominal_vth__V: Tensor
    beta__uA_per_V2: Tensor
    vth__V: Tensor

    def __init__(
        self,
        cfg: NMOSConfig,
        *,
        name: str = "",
        T__K: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self._neurox_name = name
        self.cfg = cfg
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
        nominal_beta__uA_per_V2 = nominal_mu__cm2_per_V_s * cfg.c_ox__fF_per_um2 * 0.1 * (cfg.W__um / cfg.L__um)
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
        # Sentinel fabricated buffers — :meth:`fabricate` overwrites
        # them.  Initialised to fresh clones of the nominals (no
        # expand) so ``.to(device)`` migrates them cleanly even
        # pre-fabricate.
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

        # Fabricate mismatch sigmas (Pelgrom area scaling)
        nominal_isqrt_area__per_um = 1.0 / math.sqrt(cfg.W__um * cfg.L__um)
        self.sigma_vth__V = cfg.A_vt__mV_um * 1e-3 * nominal_isqrt_area__per_um if cfg.A_vt__mV_um is not None else None
        self.sigma_beta__uA_per_V2 = (
            nominal_beta__uA_per_V2 * cfg.A_beta_relative__um * nominal_isqrt_area__per_um
            if cfg.A_beta_relative__um is not None
            else None
        )

        # --- Derived parasitic capacitances ---

        # Intrinsic channel cap (W·L·C_ox), split 50/50 to source and drain under the deep-triode partitioning.
        c_int__fF = cfg.W__um * cfg.L__um * cfg.c_ox__fF_per_um2
        self.c_gs__fF = 0.5 * c_int__fF + cfg.W__um * cfg.c_gso__fF_per_um
        self.c_gd__fF = 0.5 * c_int__fF + cfg.W__um * cfg.c_gdo__fF_per_um
        self.c_db__fF = cfg.A_D__um2 * cfg.c_j__fF_per_um2 + cfg.P_D__um * cfg.c_jsw__fF_per_um
        self.c_sb__fF = cfg.A_S__um2 * cfg.c_j__fF_per_um2 + cfg.P_S__um * cfg.c_jsw__fF_per_um

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Sample per-cell mismatch and register ``β`` / ``V_th`` internally.

        Pelgrom-style Gaussian draws use the area-derived sigmas
        ``σ_Vt = A_vt / sqrt(W · L)`` and
        ``σ_β = β · A_beta_relative / sqrt(W · L)`` cached at init.
        Each fabricated buffer is rebuilt from its **nominal**
        template via ``nominal.clone().expand(shape)``: the
        ``clone()`` produces a 1-element buffer that is decoupled
        from the nominal so a later overwrite cannot leak back, and
        the ``expand`` is a view at the requested per-cell shape.
        When the matching sigma is ``None`` the fabricated buffer
        stays a 1-element view (no full-shape memory).  When it is
        set, :func:`apply_gaussian` materialises the full-shape
        tensor via ``torch.randn_like``.  Re-callable: a subsequent
        :meth:`fabricate` overwrites the fabricated buffers from the
        unchanged nominals.

        Args:
            shape: Target per-cell shape (typically the xbar's
                ``[..., physical_col_num, row_num]`` cell layout).
        """
        beta__uA_per_V2 = self.nominal_beta__uA_per_V2.clone().expand(shape)
        if self.sigma_beta__uA_per_V2 is not None:
            beta__uA_per_V2 = apply_gaussian(beta__uA_per_V2, self.sigma_beta__uA_per_V2)
        self.register_buffer("beta__uA_per_V2", beta__uA_per_V2, persistent=False)

        vth__V = self.nominal_vth__V.clone().expand(shape)
        if self.sigma_vth__V is not None:
            vth__V = apply_gaussian(vth__V, self.sigma_vth__V)
        self.register_buffer("vth__V", vth__V, persistent=False)

    def snapshot(self, *, shape: tuple[int, ...]) -> NMOSSnapshot:
        """Return a per-VMM NMOS snapshot.

        Wraps the fabricated ``self.beta__uA_per_V2`` /
        ``self.vth__V`` buffers in a :class:`NMOSSnapshot`.  No
        dynamic noise is added today; future runtime-noise additions
        (drain shot, flicker, etc.) plug in here by perturbing the
        buffers before wrapping.  The ``shape`` argument mirrors the
        RRAM counterpart and is reserved for that future use.
        """
        del shape  # reserved — see docstring
        return NMOSSnapshot(beta__uA_per_V2=self.beta__uA_per_V2, vth__V=self.vth__V)

    def solve_dc(
        self,
        vg__V: Tensor | float,
        vd__V: Tensor | float,
        vs__V: Tensor | float,
        snapshot: NMOSSnapshot,
    ) -> NMOSDC:
        """Evaluate ``I_ds`` and its three node-partial derivatives at one operating point.

        Reads ``β`` / ``V_th`` from the per-VMM ``snapshot`` (not
        from ``self``) so the same Newton iteration sees a consistent
        set of fixed values and future dynamic noise plugs in
        through :meth:`NMOS.snapshot` without touching this body.

        The EKV-softplus form gives every quantity in terms of the
        four shared intermediates ``(V_eff,s, V_eff,d, σ_s, σ_d)``,
        computed once below.  Returning the bundle lets a single
        fused ``@torch.compile`` kernel cover all of ``ids`` + node
        partials.

        Closed-form partials (from chain rule through softplus,
        whose derivative is sigmoid):

            ∂I/∂V_g = β · (V_eff,s · σ_s − V_eff,d · σ_d)     [gm]
            ∂I/∂V_d = β · V_eff,d · σ_d                       (≥ 0)
            ∂I/∂V_s = −β · V_eff,s · σ_s                      (≤ 0)

        Args:
            vg__V: Gate voltage [V].  Tensor, or Python float for a
                fixed bias rail.
            vd__V: Drain voltage [V].  Tensor or float.
            vs__V: Source voltage [V].  Tensor or float.
            snapshot: Per-VMM NMOS snapshot from :meth:`snapshot` —
                carries ``β`` and ``V_th``.

        Returns:
            :class:`NMOSDC` with ``ids__uA`` plus the three node
            partials.  All tensors broadcast to the same shape.
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

        return NMOSDC(
            ids__uA=ids__uA,
            did_dvg__uS=did_dvg__uS,
            did_dvd__uS=did_dvd__uS,
            did_dvs__uS=did_dvs__uS,
        )
