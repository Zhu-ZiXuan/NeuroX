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

from .base import ADC, ADCConfig, AdcOperationPoint, ADCPolicy


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
            sigma.
        comparator_offset_sigma__V: Static Gaussian sigma on the
            comparator threshold [V].
        comparator_thermal_noise_sigma__V: Per-cycle Gaussian sigma
            for thermal comparator noise [V].
        e_bootstrap__fJ: Per-conversion bootstrapped sampling-switch
            overhead [fJ].
        e_constant_per_bit__fJ: Per-cycle SAR strobe / logic / control
            overhead [fJ]; charged ``bits`` times per conversion.
        leakage_per_inst__uW: Static leakage per ADC instance [uW].
        area_per_inst__um2: Silicon area per ADC instance [μm²].
    """

    # --- Topology ---
    max_bits: int

    # --- References + timing ---
    v_refs__V: tuple[float, ...]
    clk_period__ns: float

    # --- CDAC unit ---
    c_unit__fF: float

    # --- Cap mismatch (Pelgrom) ---
    cap_mismatch_sigma_relative: float

    # --- Comparator static offset ---
    comparator_offset_sigma__V: float

    # --- Comparator thermal noise (per-cycle) ---
    comparator_thermal_noise_sigma__V: float

    # --- Energy ---
    e_bootstrap__fJ: float
    e_constant_per_bit__fJ: float

    # --- PPA ---
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
        self._require_nonneg(self.cap_mismatch_sigma_relative, "cap_mismatch_sigma_relative")

    def validate_comparator(self) -> None:
        self._require_nonneg(self.comparator_offset_sigma__V, "comparator_offset_sigma__V")
        self._require_nonneg(self.comparator_thermal_noise_sigma__V, "comparator_thermal_noise_sigma__V")

    def validate_energy(self) -> None:
        self._require_nonneg(self.e_bootstrap__fJ, "e_bootstrap__fJ")
        self._require_nonneg(self.e_constant_per_bit__fJ, "e_constant_per_bit__fJ")

    def validate_ppa(self) -> None:
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")


@dataclass(frozen=True)
class McsSarAdcPolicy(ADCPolicy):
    """Per-source toggles selecting which McsSarAdc nonidealities are active.

    Attributes:
        cap_mismatch: Apply ``cap_mismatch_sigma_relative`` at fabricate time.
        comparator_offset: Apply ``comparator_offset_sigma__V`` at fabricate time.
        comparator_thermal_noise: Apply ``comparator_thermal_noise_sigma__V`` per SAR cycle.
        sampling_thermal_noise: Apply kT/C sampling thermal noise on the held top plates.
    """

    cap_mismatch: bool
    comparator_offset: bool
    comparator_thermal_noise: bool
    sampling_thermal_noise: bool


@ADC.register_key(McsSarAdcConfig)
class McsSarAdc(ADC):
    """V_cm-based (MCS) differential SAR ADC.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        name: Hierarchical instance name used by the profiler.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature [K].
    """

    nominal_c__fF: Tensor
    nominal_comparator_offset__V: Tensor
    c_p__fF: Tensor
    c_n__fF: Tensor
    comparator_offset__V: Tensor

    def __init__(
        self,
        *,
        config: McsSarAdcConfig,
        policy: McsSarAdcPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            name=name,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        if not (T__K > 0.0):
            raise ValueError(f"McsSarAdc T__K ({T__K}) must be > 0")

        self.config = config
        self.policy = policy
        self.T__K = T__K
        self.dtype = dtype

        self.comparator_noise_sigma__V = config.comparator_thermal_noise_sigma__V * math.sqrt(T__K / 300.0)

        self.n_caps = config.max_bits

        c_unit = config.c_unit__fF
        nominal_c__fF = torch.tensor(
            [c_unit] + [c_unit * (2**k) for k in range(config.max_bits - 1)],
            dtype=dtype,
        )
        self.register_buffer("nominal_c__fF", nominal_c__fF, persistent=False)
        self.register_buffer(
            "nominal_comparator_offset__V",
            torch.zeros((), dtype=dtype),
            persistent=False,
        )

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

        self._log_static()

    # --- runtime-mode introspection ---

    def available_modes(self) -> tuple[float, ...]:
        """V_ref values the configured CDAC supports, in index order."""
        return self.config.v_refs__V

    @property
    def mode_num(self) -> int:
        """Number of operating points — one per supported V_ref."""
        return len(self.config.v_refs__V)

    @property
    def max_bits(self) -> int:
        """Physical CDAC bit width — the maximum ``adc_bits`` value."""
        return self.config.max_bits

    # --- fabricate (static non-idealities) ---

    def _sample_fabricate_mismatch(self) -> None:
        """Resample independent differential cap arrays and comparator offset."""
        config = self.config
        inst_shape = self._inst_shape

        # Two independently-sampled cap arrays for the differential CDAC.
        policy = self.policy
        self.c_p__fF = apply_pelgrom_mismatch(
            self.nominal_c__fF.clone().expand(*inst_shape, self.n_caps),
            config.cap_mismatch_sigma_relative,
            unit=config.c_unit__fF,
            floor=0.1 * config.c_unit__fF,
            enabled=policy.cap_mismatch,
        )
        self.c_n__fF = apply_pelgrom_mismatch(
            self.nominal_c__fF.clone().expand(*inst_shape, self.n_caps),
            config.cap_mismatch_sigma_relative,
            unit=config.c_unit__fF,
            floor=0.1 * config.c_unit__fF,
            enabled=policy.cap_mismatch,
        )
        self.comparator_offset__V = apply_gaussian(
            self.nominal_comparator_offset__V.clone().expand(inst_shape),
            config.comparator_offset_sigma__V,
            enabled=policy.comparator_offset,
        )

    # --- ABC contract ---

    @property
    def area_per_inst__um2(self) -> float:
        """Silicon area per instance [um^2]."""
        return self.config.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per instance [uW]."""
        return self.config.leakage_per_inst__uW

    def latency_per_op__ns(self, *, adc_operation_point: AdcOperationPoint) -> float:
        """Per-conversion latency ``(adc_bits + 1) · clk_period__ns``.

        Args:
            adc_operation_point: Runtime operating point.  ``1 ≤ bits ≤ max_bits``.
        """
        bits = adc_operation_point.adc_bits
        if not (1 <= bits <= self.config.max_bits):
            raise ValueError(f"bits {bits} outside [1, {self.config.max_bits}]")
        return (bits + 1) * self.config.clk_period__ns

    # --- convert ---

    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        adc_operation_point: AdcOperationPoint,
    ) -> Tensor:
        """V_cm-based (MCS) differential SAR conversion.

        Args:
            v_pos__V: Positive-side input voltage.
            v_neg__V: Negative-side input voltage, same shape.
            adc_operation_point: Runtime operating point. ``adc_operation_point.adc_mode`` selects V_ref;
                ``adc_operation_point.adc_bits`` sets active resolution.

        Returns:
            Code tensor in ``[0, 2 ** adc_operation_point.adc_bits - 1]``.
        """
        self._validate_runtime_args(adc_operation_point)
        bits = adc_operation_point.adc_bits

        config = self.config
        v_ref__V = config.v_refs__V[adc_operation_point.adc_mode]
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
        kt__fJ = K_BOLTZMANN__J_per_K * self.T__K * 1e15
        v_p_top__V = apply_gaussian(
            v_p_top__V, torch.sqrt(kt__fJ / c_p_total__fF), enabled=self.policy.sampling_thermal_noise
        )
        v_n_top__V = apply_gaussian(
            v_n_top__V, torch.sqrt(kt__fJ / c_n_total__fF), enabled=self.policy.sampling_thermal_noise
        )

        # Sample energy: input source charges the bottom-plate caps from V_cm to V_in.
        e_p_sample__fJ = c_p_total__fF * v_pos__V * torch.clamp_min(v_pos__V - v_cm__V, 0)
        e_n_sample__fJ = c_n_total__fF * v_neg__V * torch.clamp_min(v_neg__V - v_cm__V, 0)
        e_sample__fJ = e_p_sample__fJ + e_n_sample__fJ + config.e_bootstrap__fJ

        # --- 2. MSB decision (free, no cap switch) ---
        # neg cap top to comparator Vin+, pos cap top to comparator Vin-
        last_bit = self._compare(v_pos__V=v_n_top__V, v_neg__V=v_p_top__V)
        code = last_bit.to(torch.int32)

        # --- SAR loop ---
        e_detect__fJ = torch.zeros_like(v_pos__V)
        c_diff__fF = torch.zeros_like(v_pos__V)
        for k in range(bits - 2, -1, -1):
            idx = k - bits + config.max_bits + 1
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

        e_detect__fJ = e_detect__fJ + bits * config.e_constant_per_bit__fJ

        # --- reset: dissipate residual differential charge ---
        e_reset__fJ = torch.abs(0.5 * v_ref__V**2 * c_diff__fF)

        e_dynamic__fJ = e_sample__fJ + e_detect__fJ + e_reset__fJ

        # --- final code: optional stochastic LSB jitter, clamp ---
        code = apply_lsb_jitter(
            code,
            n_bits=bits,
            enabled=self.training,
        )
        code = code.clamp(min=0, max=(1 << bits) - 1)
        self._log_dynamic(e_dynamic__fJ, self.latency_per_op__ns(adc_operation_point=adc_operation_point))
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
        v_diff__V = apply_gaussian(
            v_pos__V - v_neg__V,
            self.comparator_noise_sigma__V,
            enabled=self.policy.comparator_thermal_noise,
        )
        return v_diff__V > self.comparator_offset__V

    # --- shared helpers ---

    def _validate_runtime_args(self, adc_operation_point: AdcOperationPoint) -> None:
        """Validate per-call ``adc_operation_point``."""
        config = self.config
        bits = adc_operation_point.adc_bits
        mode = adc_operation_point.adc_mode
        if not (0 <= mode < len(config.v_refs__V)):
            raise ValueError(f"mode {mode} outside [0, {len(config.v_refs__V)})")
        if not (1 <= bits <= config.max_bits):
            raise ValueError(f"bits {bits} outside [1, {config.max_bits}]")
