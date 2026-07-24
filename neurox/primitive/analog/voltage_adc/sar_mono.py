"""Monotonic (Set-and-Down) differential SAR voltage ADC — placeholder.

See also:
    docs/reference/primitive/analog/voltage_adc/sar_mono.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.primitive.nonideality import (
    apply_gaussian,
    apply_pelgrom_mismatch,
)

from .base import DifferentialVoltageAdc, DifferentialVoltageAdcConfig, DifferentialVoltageAdcPolicy


@dataclass(frozen=True)
class SarMonoDifferentialVoltageAdcConfig(DifferentialVoltageAdcConfig):
    """Immutable design-parameter config for :class:`SarMonoDifferentialVoltageAdc`.

    Attributes:
        max_bits: Physical bit width; active array carries
            ``max_bits - 1`` binary-weighted caps.
        clk_period__ns: SAR comparator clock period; latency at
            ``bits`` active bits is ``(bits + 1) · clk_period``.
        c_unit__fF: CDAC unit capacitance.
        cap_mismatch_sigma_relative: Per-unit-cap relative Pelgrom
            σ.
        comparator_offset_sigma__V: Static comparator-threshold Gaussian
            σ.
        comparator_thermal_noise_sigma__V: Per-cycle dynamic
            comparator-noise Gaussian σ.
        e_bootstrap__fJ: Per-conversion sampling-switch overhead.
        e_compare_per_bit__fJ: Per-cycle comparator-decision energy.
        e_logic_per_bit__fJ: Per-cycle SAR-logic / register overhead.
    """

    # --- Topology ---
    max_bits: int

    # --- Timing ---
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

    def validate(self) -> None:
        super().validate()
        self.validate_topology()
        self.validate_timing()
        self.validate_cdac()
        self.validate_comparator()
        self.validate_energy()
        self.validate_ppa()

    def validate_topology(self) -> None:
        if not (self.max_bits >= 2):
            raise ValueError(f"require: max_bits ({self.max_bits}) >= 2")

    def validate_timing(self) -> None:
        self._require_pos(self.clk_period__ns, "clk_period__ns")

    def validate_cdac(self) -> None:
        self._require_pos(self.c_unit__fF, "c_unit__fF")
        self._require_non_neg(self.cap_mismatch_sigma_relative, "cap_mismatch_sigma_relative")

    def validate_comparator(self) -> None:
        self._require_non_neg(self.comparator_offset_sigma__V, "comparator_offset_sigma__V")
        self._require_non_neg(self.comparator_thermal_noise_sigma__V, "comparator_thermal_noise_sigma__V")

    def validate_energy(self) -> None:
        self._require_non_neg(self.e_bootstrap__fJ, "e_bootstrap__fJ")
        self._require_non_neg(self.e_compare_per_bit__fJ, "e_compare_per_bit__fJ")
        self._require_non_neg(self.e_logic_per_bit__fJ, "e_logic_per_bit__fJ")


@dataclass(frozen=True)
class SarMonoDifferentialVoltageAdcPolicy(DifferentialVoltageAdcPolicy):
    """Per-source toggles selecting which SarMonoDifferentialVoltageAdc nonidealities are active.

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


@DifferentialVoltageAdc.register_key(SarMonoDifferentialVoltageAdcConfig)
class SarMonoDifferentialVoltageAdc(
    DifferentialVoltageAdc[SarMonoDifferentialVoltageAdcConfig, SarMonoDifferentialVoltageAdcPolicy]
):
    """Monotonic (Set-and-Down) differential SAR voltage ADC — placeholder.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    nominal_cap_weights__fF: Tensor
    nominal_comparator_offset__V: Tensor
    c_p__fF: Tensor
    c_n__fF: Tensor
    comparator_offset__V: Tensor

    def __init__(
        self,
        *,
        config: SarMonoDifferentialVoltageAdcConfig,
        policy: SarMonoDifferentialVoltageAdcPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
        if not (T__K > 0.0):
            raise ValueError(f"SarMonoDifferentialVoltageAdc T__K ({T__K}) must be > 0")
        self._area_per_inst__um2 = config.area_per_inst__um2
        self._leakage_per_inst__uW = config.leakage_per_inst__uW
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

    # --- runtime-mode introspection ---

    @property
    def max_bits(self) -> int:
        """Physical CDAC bit width — the maximum ``bits`` value."""
        return self.config.max_bits

    def unsigned_range(self, bits: int) -> tuple[int, int]:
        """Raw offset-binary code endpoints at ``bits`` — ``(0, 2 ** bits - 1)``.

        The CDAC code count is exactly ``2 ** bits``.
        """
        if not (1 <= bits <= self.max_bits):
            raise ValueError(f"bits {bits} outside [1, {self.max_bits}]")
        return 0, (1 << bits) - 1

    def zero_offset(self, bits: int) -> int:
        """Offset-binary zero code at ``bits`` — ``2 ** (bits - 1)``."""
        if not (1 <= bits <= self.max_bits):
            raise ValueError(f"bits {bits} outside [1, {self.max_bits}]")
        return 1 << (bits - 1)

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

    # --- convert ---

    def _convert_impl(
        self,
        v_pos__V: Tensor,
        v_neg__V: Tensor,
        *,
        v_ref__V: Tensor,
        bits: int,
    ) -> Tensor:
        """Differential monotonic SAR conversion — not yet implemented."""
        del v_pos__V, v_neg__V, v_ref__V, bits
        raise NotImplementedError(
            "Differential monotonic SAR is not yet implemented; use McsSarDifferentialVoltageAdc."
        )

    # --- shared helpers ---

    def _validate_runtime_args(self, bits: int) -> None:
        """Validate the per-call bit width against the config bound."""
        config = self.config
        if not (1 <= bits <= config.max_bits):
            raise ValueError(f"bits {bits} outside [1, {config.max_bits}]")
