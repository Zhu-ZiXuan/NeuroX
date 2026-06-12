"""Monotonic (Set-and-Down) differential SAR ADC — placeholder.

See also:
    docs/dev/modules/analog/adc/sar_mono.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.nonideality import (
    apply_gaussian,
    apply_pelgrom_mismatch,
)

from .base import ADC, ADCConfig, AdcOperationPoint, ADCPolicy


@dataclass(frozen=True)
class SarAdcMonoConfig(ADCConfig):
    """Immutable design-parameter config for :class:`SarAdcMono`.

    Attributes:
        max_bits: Physical bit width; active array carries
            ``max_bits - 1`` binary-weighted caps + a dummy cap.
        v_refs: Supported reference voltages in input units;
            ``v_refs[0]`` is the calibration anchor.
        clk_period__ns: SAR comparator clock period [ns]; latency at
            ``bits`` active bits is ``(bits + 1) · clk_period``.
        c_unit__fF: CDAC unit capacitance [fF].
        cap_mismatch_sigma_relative: Per-unit-cap relative Pelgrom
            sigma.
        comparator_offset_sigma__V: Static comparator-threshold Gaussian
            sigma [V].
        comparator_thermal_noise_sigma__V: Per-cycle dynamic
            comparator-noise Gaussian sigma [V].
        e_bootstrap__fJ: Per-conversion sampling-switch overhead [fJ].
        e_compare_per_bit__fJ: Per-cycle comparator-decision energy
            [fJ].
        e_logic_per_bit__fJ: Per-cycle SAR-logic / register overhead
            [fJ].
        leakage_per_inst__uW: Static leakage per ADC instance [uW].
        area_per_inst__um2: Silicon area per ADC instance [μm²].
    """

    # --- Topology ---
    max_bits: int

    # --- References + timing ---
    v_refs: tuple[float, ...]
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
    e_compare_per_bit__fJ: float
    e_logic_per_bit__fJ: float

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
        self._require_min_length(self.v_refs, 1, "v_refs")
        for i, v in enumerate(self.v_refs):
            self._require_pos(v, f"v_refs[{i}]")

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
        self._require_nonneg(self.e_compare_per_bit__fJ, "e_compare_per_bit__fJ")
        self._require_nonneg(self.e_logic_per_bit__fJ, "e_logic_per_bit__fJ")

    def validate_ppa(self) -> None:
        self._require_nonneg(self.leakage_per_inst__uW, "leakage_per_inst__uW")
        self._require_nonneg(self.area_per_inst__um2, "area_per_inst__um2")


@dataclass(frozen=True)
class SarAdcMonoPolicy(ADCPolicy):
    """Per-source toggles selecting which SarAdcMono nonidealities are active.

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


@ADC.register_key(SarAdcMonoConfig)
class SarAdcMono(ADC):
    """Monotonic (Set-and-Down) differential SAR ADC — placeholder.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        name: Hierarchical instance name used by the profiler.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature [K].
    """

    nominal_cap_weights__fF: Tensor
    nominal_comparator_offset__V: Tensor
    c_p__fF: Tensor
    c_n__fF: Tensor
    comparator_offset__V: Tensor

    def __init__(
        self,
        *,
        config: SarAdcMonoConfig,
        policy: SarAdcMonoPolicy,
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
            raise ValueError(f"SarAdcMono T__K ({T__K}) must be > 0")
        self.config = config
        self.policy = policy
        self.T__K = T__K
        self.dtype = dtype

        n_caps = config.max_bits - 1
        nominal_cap_weights__fF = torch.tensor(
            [config.c_unit__fF * (2**k) for k in range(n_caps)],
            dtype=dtype,
        )
        self.register_buffer("nominal_cap_weights__fF", nominal_cap_weights__fF, persistent=False)
        self.register_buffer(
            "nominal_comparator_offset__V",
            torch.zeros((), dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "c_p__fF",
            self.nominal_cap_weights__fF.clone(),
            persistent=False,
        )
        self.register_buffer(
            "c_n__fF",
            self.nominal_cap_weights__fF.clone(),
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
        return self.config.v_refs

    @property
    def mode_num(self) -> int:
        """Number of operating points — one per supported V_ref."""
        return len(self.config.v_refs)

    @property
    def max_bits(self) -> int:
        """Physical CDAC bit width — the maximum ``adc_bits`` value."""
        return self.config.max_bits

    def signed_range(self, adc_bits: int) -> tuple[int, int]:
        """Canonical SAR signed-bit endpoints at ``adc_bits``.

        Mirrors :meth:`McsSarAdc.signed_range` — the CDAC code count is
        exactly ``2 ** adc_bits``.
        """
        if not (1 <= adc_bits <= self.max_bits):
            raise ValueError(f"adc_bits {adc_bits} outside [1, {self.max_bits}]")
        half = 1 << (adc_bits - 1)
        return -half, half - 1

    # --- fabricate (static non-idealities) ---

    def _sample_fabricate_mismatch(self) -> None:
        """Resample cap mismatch and comparator offset at ``self._inst_shape``."""
        config = self.config
        n_caps = config.max_bits - 1
        inst_shape = self._inst_shape

        policy = self.policy
        self.c_p__fF = apply_pelgrom_mismatch(
            self.nominal_cap_weights__fF.clone().expand(*inst_shape, n_caps),
            config.cap_mismatch_sigma_relative,
            unit=config.c_unit__fF,
            floor=0.1 * config.c_unit__fF,
            enabled=policy.cap_mismatch,
        )
        self.c_n__fF = apply_pelgrom_mismatch(
            self.nominal_cap_weights__fF.clone().expand(*inst_shape, n_caps),
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
        """Per-conversion latency at the runtime bit width.

        Args:
            adc_operation_point: Runtime operating point.  ``1 ≤ adc_operation_point.adc_bits ≤ max_bits``.
        """
        if not (1 <= adc_operation_point.adc_bits <= self.config.max_bits):
            raise ValueError(f"bits {adc_operation_point.adc_bits} outside [1, {self.config.max_bits}]")
        return (adc_operation_point.adc_bits + 1) * self.config.clk_period__ns

    # --- convert ---

    def convert(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        adc_operation_point: AdcOperationPoint,
    ) -> Tensor:
        """Differential monotonic SAR conversion — not yet implemented."""
        del v_pos__V, v_neg__V, adc_operation_point
        raise NotImplementedError("Differential monotonic SAR is not yet implemented; use McsSarAdc.")

    # --- shared helpers ---

    def _validate_runtime_args(self, adc_operation_point: AdcOperationPoint) -> None:
        """Validate per-call ``adc_operation_point``."""
        config = self.config
        if not (0 <= adc_operation_point.adc_mode < len(config.v_refs)):
            raise ValueError(f"mode {adc_operation_point.adc_mode} outside [0, {len(config.v_refs)})")
        if not (1 <= adc_operation_point.adc_bits <= config.max_bits):
            raise ValueError(f"bits {adc_operation_point.adc_bits} outside [1, {config.max_bits}]")
