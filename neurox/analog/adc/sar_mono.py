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

from .base import ADC, ADCConfig, AdcOperationPoint


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
        enable_cap_mismatch: Apply ``cap_mismatch_sigma_relative`` at
            fabricate time.
        enable_comparator_offset: Apply
            ``comparator_offset_sigma__V`` at fabricate time.
        enable_comparator_thermal_noise: Apply
            ``comparator_thermal_noise_sigma__V`` per SAR cycle.
        enable_sampling_thermal_noise: Apply kT/C sampling thermal
            noise on the held top plates.
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
    enable_cap_mismatch: bool

    # --- Comparator static offset ---
    comparator_offset_sigma__V: float
    enable_comparator_offset: bool

    # --- Comparator thermal noise (per-cycle) ---
    comparator_thermal_noise_sigma__V: float
    enable_comparator_thermal_noise: bool

    # --- Sampling thermal noise (kT/C) ---
    enable_sampling_thermal_noise: bool

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


@ADC.register_key(SarAdcMonoConfig)
class SarAdcMono(ADC):
    """Monotonic (Set-and-Down) differential SAR ADC — placeholder.

    Args:
        cfg: Concrete configuration dataclass.
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
        cfg: SarAdcMonoConfig,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(cfg=cfg, name=name, inst_shape=inst_shape, dtype=dtype, T__K=T__K)
        if not (T__K > 0.0):
            raise ValueError(f"SarAdcMono T__K ({T__K}) must be > 0")
        self.cfg = cfg
        self.T__K = T__K
        self.dtype = dtype

        n_caps = cfg.max_bits - 1
        nominal_cap_weights__fF = torch.tensor(
            [cfg.c_unit__fF * (2**k) for k in range(n_caps)],
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
        return self.cfg.v_refs

    @property
    def mode_num(self) -> int:
        """Number of operating points — one per supported V_ref."""
        return len(self.cfg.v_refs)

    @property
    def max_bits(self) -> int:
        """Physical CDAC bit width — the maximum ``adc_bits`` value."""
        return self.cfg.max_bits

    # --- fabricate (static non-idealities) ---

    def _sample_fabricate_mismatch(self) -> None:
        """Resample cap mismatch and comparator offset at ``self._inst_shape``."""
        cfg = self.cfg
        n_caps = cfg.max_bits - 1
        inst_shape = self._inst_shape

        self.c_p__fF = apply_pelgrom_mismatch(
            self.nominal_cap_weights__fF.clone().expand(*inst_shape, n_caps),
            cfg.cap_mismatch_sigma_relative,
            unit=cfg.c_unit__fF,
            floor=0.1 * cfg.c_unit__fF,
            enabled=cfg.enable_cap_mismatch,
        )
        self.c_n__fF = apply_pelgrom_mismatch(
            self.nominal_cap_weights__fF.clone().expand(*inst_shape, n_caps),
            cfg.cap_mismatch_sigma_relative,
            unit=cfg.c_unit__fF,
            floor=0.1 * cfg.c_unit__fF,
            enabled=cfg.enable_cap_mismatch,
        )
        self.comparator_offset__V = apply_gaussian(
            self.nominal_comparator_offset__V.clone().expand(inst_shape),
            cfg.comparator_offset_sigma__V,
            enabled=cfg.enable_comparator_offset,
        )

    # --- ABC contract ---

    @property
    def area_per_inst__um2(self) -> float:
        """Silicon area per instance [um^2]."""
        return self.cfg.area_per_inst__um2

    @property
    def leakage_per_inst__uW(self) -> float:
        """Static leakage per instance [uW]."""
        return self.cfg.leakage_per_inst__uW

    def latency_per_op__ns(self, *, adc_operation_point: AdcOperationPoint) -> float:
        """Per-conversion latency at the runtime bit width.

        Args:
            adc_operation_point: Runtime operating point.  ``1 ≤ adc_operation_point.adc_bits ≤ max_bits``.
        """
        if not (1 <= adc_operation_point.adc_bits <= self.cfg.max_bits):
            raise ValueError(f"bits {adc_operation_point.adc_bits} outside [1, {self.cfg.max_bits}]")
        return (adc_operation_point.adc_bits + 1) * self.cfg.clk_period__ns

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
        cfg = self.cfg
        if not (0 <= adc_operation_point.adc_mode < len(cfg.v_refs)):
            raise ValueError(f"mode {adc_operation_point.adc_mode} outside [0, {len(cfg.v_refs)})")
        if not (1 <= adc_operation_point.adc_bits <= cfg.max_bits):
            raise ValueError(f"bits {adc_operation_point.adc_bits} outside [1, {cfg.max_bits}]")
