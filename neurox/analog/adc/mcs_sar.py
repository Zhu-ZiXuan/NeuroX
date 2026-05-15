"""V_cm-based (Merged Capacitor Switching, MCS) differential SAR ADC.

Implements the differential MCS topology with bottom-plate sampling:

1. **Sample**: bottom plates of two independent CDAC arrays track the
   positive / negative inputs; the top plates float.  On release the
   bottom plates snap to V_cm, leaving each top plate at ``V_ref - V_in``.
2. **MSB decision**: free comparison of the two top-plate voltages
   (with static offset and per-cycle thermal noise).
3. **SAR loop**: each cycle picks one cap on each leg and switches it
   from V_cm to V_ref or to GND according to the previous bit, moving
   the top plates by ``± V_cm · C_k / C_total``.

Dynamic energy (fJ = fF · V²):

* **Sample**: input source charges bottom-plate caps from V_cm to V_in.
* **SAR per-cycle**: standard MCS formula
  ``½ · V_ref² · C_k · (1 - C_k / C_total)``, charged on whichever
  leg's cap moves toward V_ref.
* **Reset**: residual differential charge dumped at the end of
  conversion, ``|½ · V_ref² · ΔC|`` where ΔC is the running
  weighted-mismatch sum.
* **Bootstrap / comparator / logic** lump-sum constants from the
  config — front-end and digital overhead independent of the cap
  network.

Static and dynamic non-idealities
---------------------------------

* **Static** (sampled at :meth:`fabricate`): per-cap Pelgrom mismatch
  drawn independently for the positive and negative legs; comparator
  threshold offset.
* **Dynamic** (sampled per :meth:`convert`): kT/C sampling noise on
  the held top plates, per-cycle comparator thermal noise, optional
  LSB stochastic-rounding jitter on the final code.

Multi-mode support
------------------

Per-instance config: ``max_bits`` (physical CDAC depth) and
``v_refs__V`` (supported reference voltages, in V).  Per-call
selectors: ``mode`` (V_ref index) and ``bits`` (active bit width).
When ``bits < max_bits`` the SAR loop engages only the top
``bits - 1`` caps; the smallest caps stay idle.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.nonideality import (
    apply_gaussian,
    apply_lsb_jitter,
    apply_pelgrom_mismatch,
)
from neurox.common.physical_constant import K_BOLTZMANN__J_per_K
from neurox.common.quant import use_stochastic

from .base import ADC


@dataclass(frozen=True)
class McsSarAdcConfig:
    """Immutable design-parameter config for :class:`McsSarAdc`.

    Attributes:
        max_bits: Physical bit width.  Number of CDAC stages laid out
            in silicon.  The MSB cap is dropped; the active cap array
            has ``max_bits - 1`` binary-weighted caps + a dummy cap.
        v_refs__V: Supported reference voltages [V], in strictly
            decreasing order — ``v_refs__V[0]`` is the maximum (the
            calibration anchor) and is the largest entry; lower
            indices correspond to coarser dynamic ranges.
        clk_period__ns: SAR comparator clock period.  Latency at
            ``bits`` active bits is ``(bits + 1) · clk_period``.
        c_unit__fF: Unit capacitance of the binary-weighted CDAC.
        cap_mismatch_sigma_relative: Per-unit-cap relative-σ Pelgrom
            sigma (eg. ``0.01`` for 1% matching).  Sigma per cap
            scales as ``sqrt(C_k / C_unit)``.  ``None`` = no mismatch
            (ideal CDAC).
        comparator_offset_sigma__V: Static Gaussian sigma on the
            comparator threshold [V].  ``None`` = no static offset.
        comparator_noise_sigma__V: Dynamic per-cycle Gaussian
            comparator noise sigma [V].  ``None`` = no comparator
            noise.
        kt_c_noise_enabled: When ``True``, sampling adds a Gaussian
            of σ = sqrt(k_B · T / C_total) to the held top plates.
        e_bootstrap__fJ: Constant per-conversion energy charged to
            the bootstrapped sampling switches (front-end overhead
            independent of the cap network).  Set to ``0.0`` to
            disable.
        e_constant_per_bit__fJ: Per-cycle constant energy lumping
            comparator strobe overhead and the SAR digital logic /
            register / control path.  Charged on every cycle
            including the MSB decision — ``bits`` cycles per
            conversion.  Set to ``0.0`` to disable.
        leakage_per_inst__uW: Static leakage per ADC instance.
        area_per_inst__um2: Silicon area per ADC instance.

    Note:
        Operating temperature is intentionally **not** a config field
        — it is a changeable operating-state quantity passed to the
        ADC class as an init arg.
    """

    max_bits: int
    v_refs__V: tuple[float, ...]

    clk_period__ns: float

    c_unit__fF: float
    cap_mismatch_sigma_relative: float | None

    comparator_offset_sigma__V: float | None
    comparator_thermal_noise_sigma__V: float | None

    enable_thermal_noise: bool

    e_bootstrap__fJ: float
    e_constant_per_bit__fJ: float

    leakage_per_inst__uW: float
    area_per_inst__um2: float

    def __post_init__(self) -> None:
        if self.max_bits < 2:
            raise ValueError(
                f"McsSarAdcConfig.max_bits ({self.max_bits}) must be >= 2 "
                f"(MSB-free design needs at least one active cap)"
            )
        if not self.v_refs__V:
            raise ValueError("McsSarAdcConfig.v_refs must contain at least one V_ref")
        for i, v in enumerate(self.v_refs__V):
            if not (v > 0.0):
                raise ValueError(f"McsSarAdcConfig.v_refs[{i}] ({v}) must be > 0")
        for i in range(1, len(self.v_refs__V)):
            if not (self.v_refs__V[i] < self.v_refs__V[i - 1]):
                raise ValueError(
                    f"McsSarAdcConfig.v_refs__V must be strictly decreasing "
                    f"(largest at index 0); got v_refs__V[{i - 1}]={self.v_refs__V[i - 1]} "
                    f"and v_refs__V[{i}]={self.v_refs__V[i]}"
                )
        if self.clk_period__ns <= 0:
            raise ValueError(f"McsSarAdcConfig.clk_period__ns ({self.clk_period__ns}) must be > 0")
        if self.c_unit__fF <= 0:
            raise ValueError(f"McsSarAdcConfig.c_unit__fF ({self.c_unit__fF}) must be > 0")
        if self.e_bootstrap__fJ < 0:
            raise ValueError(f"McsSarAdcConfig.e_bootstrap__fJ ({self.e_bootstrap__fJ}) must be >= 0")
        if self.e_constant_per_bit__fJ < 0:
            raise ValueError(f"McsSarAdcConfig.e_constant_per_bit__fJ ({self.e_constant_per_bit__fJ}) must be >= 0")


class McsSarAdc(ADC):
    """V_cm-based (MCS) differential SAR ADC.

    Args:
        config: Topology configuration.
        T__K: Operating temperature in Kelvin.  Drives the kT/C
            sampling-noise model.  Per project convention temperature
            is an operating-state init arg, not a config (design)
            field.  Must be ``> 0``.
        stochastic: Per-instance switch for additional LSB-jitter
            stochastic rounding on top of the SAR pipeline.  ``None``
            (default) follows ``module.training``; ``True`` / ``False``
            force on / off.
        dtype: Float dtype for internal voltage arithmetic.
    """

    # --- static buffers --- #
    # ``nominal_c__fF`` is the design-time LSB-first cap template
    #     [C_unit, C_unit, 2·C_unit, 4·C_unit, ..., 2^(max_bits-2)·C_unit]
    # (length ``max_bits``).  Index 0 is the dummy unit cap; index
    # ``i`` for ``i ≥ 1`` is the binary-weighted cap of weight
    # ``2^(i-1) · C_unit``.  The MSB cap (weight 2^(max_bits-1))
    # is intentionally absent — the differential MCS topology
    # resolves the MSB by free comparison, not by switching a cap.
    # The buffer is the **nominal** (ideal) template — constant
    # across the ADC's lifetime and read by :meth:`fabricate` on
    # every call to rebuild the per-leg fabricated tensors.
    #
    # ``nominal_comparator_offset__V`` is the 0-d nominal comparator
    # threshold (always zero).  The fabricated
    # ``comparator_offset__V`` is sampled from
    # ``cfg.comparator_offset_sigma__V`` over the requested instance
    # shape; when ``sigma`` is ``None`` the fabricated buffer stays a
    # 1-element view of the nominal.
    #
    # ``c_p__fF`` / ``c_n__fF`` are the two independently-sampled cap
    # arrays of the differential CDAC (positive / negative leg);
    # ``comparator_offset__V`` is the static comparator threshold
    # offset.  All three are rebuilt from their nominal templates on
    # every :meth:`fabricate` call.  Pre-fabricate sentinels are
    # fresh clones of the nominals so ``.to(device)`` migrates
    # cleanly even before fabricate.
    nominal_c__fF: Tensor
    nominal_comparator_offset__V: Tensor
    c_p__fF: Tensor
    c_n__fF: Tensor
    comparator_offset__V: Tensor

    def __init__(
        self,
        cfg: McsSarAdcConfig,
        *,
        name: str = "",
        T__K: float,
        stochastic: bool | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(name=name)

        self.cfg = cfg
        self.T__K: float = T__K
        self.stochastic: bool | None = stochastic

        # Per-cycle comparator thermal-noise sigma, temperature-scaled
        # once at init (σ ∝ sqrt(T) for thermal noise; the config sigma
        # is anchored at 300 K).  Stored as a Python float (or None when
        # comparator noise is disabled) so the SAR loop only pays the
        # cheap branch on the optional path.
        self.comparator_noise_sigma__V: float | None = (
            cfg.comparator_thermal_noise_sigma__V * math.sqrt(T__K / 300.0)
            if cfg.comparator_thermal_noise_sigma__V is not None
            else None
        )

        # Cap array length = 1 dummy + (max_bits - 1) binary-weighted
        # caps; the MSB cap (weight 2^(max_bits-1)) is intentionally
        # absent.  Layout (LSB-first, index 0 = dummy):
        #     [C_unit, C_unit, 2·C_unit, 4·C_unit, ..., 2^(max_bits-2)·C_unit]
        # The dummy and the binary-weighted caps live in the same
        # tensor so ``c_total = c_p__fF.sum(dim=-1)`` is the full
        # physical array capacitance (= ``2^(max_bits-1) · C_unit``).
        # Convert-time access into the binary-weighted sub-ladder
        # skips index 0 (the dummy).
        self.n_caps = cfg.max_bits

        c_unit = cfg.c_unit__fF
        nominal_c__fF = torch.tensor(
            [c_unit] + [c_unit * (2**k) for k in range(cfg.max_bits - 1)],
            dtype=dtype,
        )
        self.register_buffer("nominal_c__fF", nominal_c__fF, persistent=False)
        self.register_buffer(
            "nominal_comparator_offset__V",
            torch.zeros((), dtype=dtype),
            persistent=False,
        )

        # Sentinel fabricated buffers — :meth:`fabricate` overwrites
        # them with shape-dependent tensors.  Initialised to fresh
        # clones of the nominals (no expand) so ``.to(device)``
        # migrates cleanly even pre-fabricate.
        self.register_buffer(
            "c_p__fF",
            self.nominal_c__fF.clone(),
            persistent=False,
        )
        self.register_buffer(
            "c_n__fF",
            self.nominal_c__fF.clone(),
            persistent=False,
        )
        self.register_buffer(
            "comparator_offset__V",
            self.nominal_comparator_offset__V.clone(),
            persistent=False,
        )

    # --- runtime-mode introspection --- #

    def available_modes(self) -> tuple[float, ...]:
        """V_ref values the configured CDAC supports, in index order."""
        return self.cfg.v_refs__V

    def max_bits(self) -> int:
        """Physical CDAC bit width — the maximum active ``bits`` value."""
        return self.cfg.max_bits

    # --- fabricate (static non-idealities) --- #

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Sample the static per-instance non-idealities and store them.

        Also records the SAR-ADC instance count for profiler static
        aggregation; the count is ``prod(shape)``.

        Two independent Pelgrom-mismatched cap arrays are rebuilt
        from :attr:`nominal_c__fF` via ``clone().expand(*shape,
        n_caps)`` (one clone per leg so the legs have independent
        storage even on the no-mismatch path), plus the static
        comparator threshold offset rebuilt from
        :attr:`nominal_comparator_offset__V`.  Re-callable: a
        subsequent :meth:`fabricate` always restarts from the
        unchanged nominals.

        Args:
            shape: Per-instance prefix shape.  ``()`` registers a
                single shared instance state (one ADC).  A larger
                shape, e.g. ``(n_columns,)``, registers per-column
                independent fabrications.
        """
        cfg = self.cfg

        # Two independently-sampled cap arrays for the differential
        # CDAC.  Each leg starts as its own ``clone().expand(...)``
        # of the nominal template so the no-mismatch path keeps two
        # distinct n_caps-element-storage views (no aliasing).
        c_p__fF = self.nominal_c__fF.clone().expand(*shape, self.n_caps)
        c_p__fF = apply_pelgrom_mismatch(
            c_p__fF,
            cfg.cap_mismatch_sigma_relative,
            unit=cfg.c_unit__fF,
            floor=0.1 * cfg.c_unit__fF,
        )
        c_n__fF = self.nominal_c__fF.clone().expand(*shape, self.n_caps)
        c_n__fF = apply_pelgrom_mismatch(
            c_n__fF,
            cfg.cap_mismatch_sigma_relative,
            unit=cfg.c_unit__fF,
            floor=0.1 * cfg.c_unit__fF,
        )

        # Static per-instance comparator threshold offset
        # (state-independent).  Starts as a 1-element view of the
        # nominal; ``apply_gaussian`` materialises only when a sigma
        # is configured.
        comparator_offset__V = self.nominal_comparator_offset__V.clone().expand(shape)
        if cfg.comparator_offset_sigma__V is not None:
            comparator_offset__V = apply_gaussian(comparator_offset__V, cfg.comparator_offset_sigma__V)

        self.register_buffer("c_p__fF", c_p__fF, persistent=False)
        self.register_buffer("c_n__fF", c_n__fF, persistent=False)
        self.register_buffer("comparator_offset__V", comparator_offset__V, persistent=False)

        # Profiler static aggregation: record per-ADC-instance count.
        self._record_inst_count(shape)

    # --- ABC contract --- #

    @property
    def area_per_inst__um2(self) -> float:
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        return self.cfg.leakage_per_inst__uW

    def latency_per_op__ns(self, *, bits: int) -> float:
        """Per-conversion latency at the runtime bit width.

        SAR latency is V_ref-independent, so ``mode`` is not part of
        the signature.

        Args:
            bits: Active bit width — required.  ``1 ≤ bits ≤ max_bits``.

        Returns:
            ``(bits + 1) · clk_period__ns``.
        """
        if not (1 <= bits <= self.cfg.max_bits):
            raise ValueError(f"bits {bits} outside [1, {self.cfg.max_bits}]")
        return (bits + 1) * self.cfg.clk_period__ns

    # --- convert --- #

    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        mode: int,
        bits: int,
    ) -> Tensor:
        """V_cm-based (MCS) differential SAR conversion.

        Args:
            v_pos__V: Positive-side input voltage.  Shape: arbitrary.
            v_neg__V: Negative-side input voltage.  Same shape.
            mode: V_ref index — required.
            bits: Active bit width — required.  ``1 ≤ bits ≤ max_bits``.

        Returns:
            Code tensor in ``[0, 2 ** bits - 1]`` broadcast to the
            input shape.  Dtype is derived from the inputs / fabricated
            buffers.  Dynamic energy and latency emit through the
            profiler side channel.
        """
        self._validate_runtime_args(mode, bits)

        cfg = self.cfg
        v_ref__V = cfg.v_refs__V[mode]
        v_cm__V = 0.5 * v_ref__V

        c_p__fF = self.c_p__fF
        c_n__fF = self.c_n__fF
        c_p_total__fF = c_p__fF.sum(dim=-1)
        c_n_total__fF = c_n__fF.sum(dim=-1)

        # --- 1. sample and hold ---
        # sample: bottom (drive): V_in, top (drive): V_cm
        # hold: bottom (drive): V_cm, top (float): 2*V_cm-V_in
        v_p_top__V = 2 * v_cm__V - v_pos__V
        v_n_top__V = 2 * v_cm__V - v_neg__V

        # sample thermal noise: sigma_V² = k_B · T / C_total.
        if cfg.enable_thermal_noise:
            kt__fJ = K_BOLTZMANN__J_per_K * self.T__K * 1e15
            v_p_top__V = apply_gaussian(v_p_top__V, torch.sqrt(kt__fJ / c_p_total__fF))
            v_n_top__V = apply_gaussian(v_n_top__V, torch.sqrt(kt__fJ / c_n_total__fF))

        # Sample energy: input source charges the bottom-plate caps from V_cm to V_in.
        e_p_sample__fJ = c_p_total__fF * v_pos__V * torch.clamp_min(v_pos__V - v_cm__V, 0)
        e_n_sample__fJ = c_n_total__fF * v_neg__V * torch.clamp_min(v_neg__V - v_cm__V, 0)
        e_sample__fJ = e_p_sample__fJ + e_n_sample__fJ + cfg.e_bootstrap__fJ

        # --- 2. MSB decision (free, no cap switch) ---
        # neg cap top to comparator Vin+, pos cap top to comparator Vin-
        last_bit = self._compare(v_pos__V=v_n_top__V, v_neg__V=v_p_top__V)
        code = last_bit.to(torch.int32)

        # --- SAR loop ---
        e_detect__fJ = torch.zeros_like(v_pos__V)
        c_diff__fF = torch.zeros_like(v_pos__V)
        for k in range(bits - 2, -1, -1):
            idx = k - bits + cfg.max_bits + 1
            c_p_k__fF = c_p__fF[..., idx]
            c_n_k__fF = c_n__fF[..., idx]
            # cap switch
            v_p_step__V = v_cm__V * c_p_k__fF / c_p_total__fF
            v_n_step__V = v_cm__V * c_n_k__fF / c_n_total__fF
            v_p_top__V = torch.where(last_bit, v_p_top__V + v_p_step__V, v_p_top__V - v_p_step__V)
            v_n_top__V = torch.where(last_bit, v_n_top__V - v_n_step__V, v_n_top__V + v_n_step__V)
            # switch energy
            c_p_eq__fF = c_p_k__fF * (1 - c_p_k__fF / c_p_total__fF)
            c_n_eq__fF = c_n_k__fF * (1 - c_n_k__fF / c_n_total__fF)
            e_detect__fJ = e_detect__fJ + 0.5 * v_ref__V**2 * torch.where(last_bit, c_p_eq__fF, c_n_eq__fF)
            # collect c_diff for reset energy
            c_diff__fF = c_diff__fF + torch.where(last_bit, -0.5, 0.5) * (c_p_k__fF - c_n_k__fF)
            # detect current bit
            # neg cap top to comparator Vin+, pos cap top to comparator Vin-
            last_bit = self._compare(v_pos__V=v_n_top__V, v_neg__V=v_p_top__V)
            code = (code << 1) | last_bit.to(torch.int32)

        e_detect__fJ = e_detect__fJ + bits * cfg.e_constant_per_bit__fJ

        # --- reset: dissipate residual differential charge ---
        e_reset__fJ = torch.abs(0.5 * v_ref__V**2 * c_diff__fF)

        e_dynamic__fJ = e_sample__fJ + e_detect__fJ + e_reset__fJ

        # --- final code: optional stochastic LSB jitter, clamp ---
        code = apply_lsb_jitter(
            code,
            n_bits=bits,
            enabled=use_stochastic(training=self.training, override=self.stochastic),
        )
        code = code.clamp(min=0, max=(1 << bits) - 1)
        self._log_dynamic(e_dynamic__fJ, self.latency_per_op__ns(bits=bits))
        return code

    def _compare(self, v_pos__V: Tensor, v_neg__V: Tensor) -> Tensor:
        """Strobe the differential comparator and return its bool output.

        Pipeline: per-cycle thermal noise (skipped when
        :attr:`comparator_noise_sigma__V` is ``None``) is added on
        differential voltage, then the static
        comparator threshold offset (a per-instance buffer drawn at
        :meth:`fabricate`) shifts the decision boundary:

            last_bit = (v_pos_noisy - v_neg_noisy) > comparator_offset__V

        Args:
            v_pos__V: Positive-side top-plate voltage.
            v_neg__V: Negative-side top-plate voltage.

        Returns:
            Bool tensor; ``True`` indicates the positive leg won.
            Shape matches the inputs.
        """
        sigma__V = self.comparator_noise_sigma__V
        v_diff__V = v_pos__V - v_neg__V
        if sigma__V is not None:
            v_diff__V = apply_gaussian(v_diff__V, sigma__V)
        return v_diff__V > self.comparator_offset__V

    # --- shared helpers --- #

    def _validate_runtime_args(self, mode: int, bits: int) -> None:
        """Validate per-call ``(mode, bits)``."""
        cfg = self.cfg
        if not (0 <= mode < len(cfg.v_refs__V)):
            raise ValueError(f"mode {mode} outside [0, {len(cfg.v_refs__V)})")
        if not (1 <= bits <= cfg.max_bits):
            raise ValueError(f"bits {bits} outside [1, {cfg.max_bits}]")
