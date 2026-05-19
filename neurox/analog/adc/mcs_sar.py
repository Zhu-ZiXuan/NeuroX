"""V_cm-based (Merged Capacitor Switching, MCS) differential SAR ADC.

See also:
    docs/dev/modules/analog/adc/mcs_sar.md
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

from .base import ADC, ADCConfig


@dataclass(frozen=True)
class McsSarAdcConfig(ADCConfig):
    """Immutable design-parameter config for :class:`McsSarAdc`.

    Attributes:
        max_bits: Physical bit width; active array carries
            ``max_bits - 1`` binary-weighted caps + a dummy cap
            (MSB-free design).
        v_refs__V: Supported reference voltages [V], strictly
            decreasing — ``v_refs__V[0]`` is the calibration anchor.
        clk_period__ns: SAR comparator clock period [ns]; latency at
            ``bits`` active bits is ``(bits + 1) · clk_period``.
        c_unit__fF: CDAC unit capacitance [fF].
        cap_mismatch_sigma_relative: Per-unit-cap relative Pelgrom
            sigma. ``None`` = ideal CDAC.
        comparator_offset_sigma__V: Static Gaussian sigma on the
            comparator threshold [V]. ``None`` skips static offset.
        comparator_thermal_noise_sigma__V: Per-cycle Gaussian sigma
            for thermal comparator noise [V]. ``None`` skips dynamic
            comparator noise.
        enable_thermal_noise: When ``True``, sampling adds
            ``σ = √(k_B · T / C_total)`` to the held top plates.
        e_bootstrap__fJ: Per-conversion bootstrapped sampling-switch
            overhead [fJ].
        e_constant_per_bit__fJ: Per-cycle SAR strobe / logic / control
            overhead [fJ]; charged ``bits`` times per conversion.
        leakage_per_inst__uW: Static leakage per ADC instance [uW].
        area_per_inst__um2: Silicon area per ADC instance [μm²].
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

    def validate(self) -> None:
        super().validate()
        self.validate_topology()
        self.validate_refs()
        self.validate_timing()
        self.validate_cdac()
        self.validate_comparator()
        self.validate_energy()
        self.validate_ppa()

    def validate_topology(self) -> None:
        if not (self.max_bits >= 2):
            raise ValueError(f"require: max_bits ({self.max_bits}) >= 2")

    def validate_refs(self) -> None:
        self._require_min_length(self.v_refs__V, 1, "v_refs__V")
        for i, v in enumerate(self.v_refs__V):
            self._require_pos(v, f"v_refs__V[{i}]")
        self._require_strictly_decreasing(self.v_refs__V, "v_refs__V")

    def validate_timing(self) -> None:
        self._require_pos(self.clk_period__ns, "clk_period__ns")

    def validate_cdac(self) -> None:
        self._require_pos(self.c_unit__fF, "c_unit__fF")
        self._require_nonneg_or_none(self.cap_mismatch_sigma_relative, "cap_mismatch_sigma_relative")

    def validate_comparator(self) -> None:
        self._require_nonneg_or_none(self.comparator_offset_sigma__V, "comparator_offset_sigma__V")
        self._require_nonneg_or_none(self.comparator_thermal_noise_sigma__V, "comparator_thermal_noise_sigma__V")

    def validate_energy(self) -> None:
        self._require_nonneg(self.e_bootstrap__fJ, "e_bootstrap__fJ")
        self._require_nonneg(self.e_constant_per_bit__fJ, "e_constant_per_bit__fJ")

    def validate_ppa(self) -> None:
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")


@ADC.register_config(McsSarAdcConfig)
class McsSarAdc(ADC):
    """V_cm-based (MCS) differential SAR ADC.

    Args:
        cfg: Topology configuration.
        name: Hierarchical profiler name (must be supplied explicitly).
        dtype: Floating-point dtype for internal voltage arithmetic.
        T__K: Operating temperature in Kelvin.  Drives the kT/C
            sampling-noise model. Must be ``> 0``.
    """

    # ``nominal_c__fF``: LSB-first cap template
    #   ``[C_unit, C_unit, 2·C_unit, ..., 2^(max_bits-2)·C_unit]`` (length ``max_bits``).
    # Index 0 is the dummy unit cap; index ``i ≥ 1`` is weight ``2^(i-1) · C_unit``.
    # ``c_p__fF`` / ``c_n__fF`` are the per-leg fabricated arrays; ``comparator_offset__V``
    # is the fabricated static threshold offset.
    nominal_c__fF: Tensor
    nominal_comparator_offset__V: Tensor
    c_p__fF: Tensor
    c_n__fF: Tensor
    comparator_offset__V: Tensor

    def __init__(
        self,
        *,
        cfg: McsSarAdcConfig,
        name: str,
        T__K: float,
        dtype: torch.dtype,
        stochastic: bool | None,
    ) -> None:
        super().__init__(cfg=cfg, name=name, T__K=T__K, dtype=dtype, stochastic=stochastic)
        if not (T__K > 0.0):
            raise ValueError(f"McsSarAdc T__K ({T__K}) must be > 0")

        self.cfg = cfg
        self.T__K: float = T__K
        self.dtype = dtype
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

    # --- runtime-mode introspection ---

    def available_modes(self) -> tuple[float, ...]:
        """V_ref values the configured CDAC supports, in index order."""
        return self.cfg.v_refs__V

    def max_bits(self) -> int:
        """Physical CDAC bit width — the maximum active ``bits`` value."""
        return self.cfg.max_bits

    # --- fabricate (static non-idealities) ---

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Sample static per-instance state over ``shape`` (re-callable).

        Args:
            shape: Per-instance fabrication shape.
        """
        cfg = self.cfg

        # Two independently-sampled cap arrays for the differential CDAC.
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

        comparator_offset__V = self.nominal_comparator_offset__V.clone().expand(shape)
        if cfg.comparator_offset_sigma__V is not None:
            comparator_offset__V = apply_gaussian(comparator_offset__V, cfg.comparator_offset_sigma__V)

        self.register_buffer("c_p__fF", c_p__fF, persistent=False)
        self.register_buffer("c_n__fF", c_n__fF, persistent=False)
        self.register_buffer("comparator_offset__V", comparator_offset__V, persistent=False)

        self._record_inst_count(shape)

    # --- ABC contract ---

    @property
    def area_per_inst__um2(self) -> float:
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        return self.cfg.leakage_per_inst__uW

    def latency_per_op__ns(self, *, bits: int) -> float:
        """Per-conversion latency ``(bits + 1) · clk_period__ns``.

        Args:
            bits: Active bit width, ``1 ≤ bits ≤ max_bits``.
        """
        if not (1 <= bits <= self.cfg.max_bits):
            raise ValueError(f"bits {bits} outside [1, {self.cfg.max_bits}]")
        return (bits + 1) * self.cfg.clk_period__ns

    # --- convert ---

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
            v_pos__V: Positive-side input voltage.
            v_neg__V: Negative-side input voltage, same shape.
            mode: V_ref index.
            bits: Active bit width, ``1 ≤ bits ≤ max_bits``.

        Returns:
            Code tensor in ``[0, 2 ** bits - 1]``.
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
        """Strobe the differential comparator.

        Adds per-cycle thermal noise to the differential voltage and
        compares against ``comparator_offset__V``.

        Args:
            v_pos__V: Positive-side top-plate voltage.
            v_neg__V: Negative-side top-plate voltage.

        Returns:
            Bool tensor; ``True`` means the positive leg won.
        """
        sigma__V = self.comparator_noise_sigma__V
        v_diff__V = v_pos__V - v_neg__V
        if sigma__V is not None:
            v_diff__V = apply_gaussian(v_diff__V, sigma__V)
        return v_diff__V > self.comparator_offset__V

    # --- shared helpers ---

    def _validate_runtime_args(self, mode: int, bits: int) -> None:
        """Validate per-call ``(mode, bits)``."""
        cfg = self.cfg
        if not (0 <= mode < len(cfg.v_refs__V)):
            raise ValueError(f"mode {mode} outside [0, {len(cfg.v_refs__V)})")
        if not (1 <= bits <= cfg.max_bits):
            raise ValueError(f"bits {bits} outside [1, {cfg.max_bits}]")
