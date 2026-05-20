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

from .base import ADC, ADCConfig


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
        cfg: Topology configuration.
        name: Hierarchical profiler name (must be supplied explicitly).
        dtype: Floating-point dtype for internal voltage arithmetic.
        T__K: Operating temperature in Kelvin.  Drives the kT/C
            sampling-noise model. Must be ``> 0``.
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
        T__K: float,
        dtype: torch.dtype,
        stochastic: bool | None,
    ) -> None:
        super().__init__(cfg=cfg, name=name, T__K=T__K, dtype=dtype, stochastic=stochastic)
        if not (T__K > 0.0):
            raise ValueError(f"SarAdcMono T__K ({T__K}) must be > 0")
        self.cfg = cfg
        self.T__K = T__K
        self.dtype = dtype
        self.stochastic = stochastic

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

    # --- runtime-mode introspection ---

    def available_modes(self) -> tuple[float, ...]:
        """V_ref values the configured CDAC supports, in index order."""
        return self.cfg.v_refs

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
        n_caps = cfg.max_bits - 1

        c_p__fF = apply_pelgrom_mismatch(
            self.nominal_cap_weights__fF.clone().expand(*shape, n_caps),
            cfg.cap_mismatch_sigma_relative,
            unit=cfg.c_unit__fF,
            floor=0.1 * cfg.c_unit__fF,
            enabled=cfg.enable_cap_mismatch,
        )
        c_n__fF = apply_pelgrom_mismatch(
            self.nominal_cap_weights__fF.clone().expand(*shape, n_caps),
            cfg.cap_mismatch_sigma_relative,
            unit=cfg.c_unit__fF,
            floor=0.1 * cfg.c_unit__fF,
            enabled=cfg.enable_cap_mismatch,
        )

        comparator_offset__V = apply_gaussian(
            self.nominal_comparator_offset__V.clone().expand(shape),
            cfg.comparator_offset_sigma__V,
            enabled=cfg.enable_comparator_offset,
        )

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
        """Per-conversion latency at the runtime bit width.

        Args:
            bits: Active bit width — required.  ``1 ≤ bits ≤ max_bits``.
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
        """Differential monotonic SAR conversion — not yet implemented."""
        del v_pos__V, v_neg__V, mode, bits
        raise NotImplementedError("Differential monotonic SAR is not yet implemented; use McsSarAdc.")

    # --- shared helpers ---

    def _validate_runtime_args(self, mode: int, bits: int) -> None:
        """Validate per-call ``(mode, bits)``."""
        cfg = self.cfg
        if not (0 <= mode < len(cfg.v_refs)):
            raise ValueError(f"mode {mode} outside [0, {len(cfg.v_refs)})")
        if not (1 <= bits <= cfg.max_bits):
            raise ValueError(f"bits {bits} outside [1, {cfg.max_bits}]")
