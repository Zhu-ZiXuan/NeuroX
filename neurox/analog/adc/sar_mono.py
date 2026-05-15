"""Monotonic (Set-and-Down) differential SAR ADC — placeholder.

Liu et al. *Set-and-Down* / monotonic switching: the larger top plate
drops by ``V_ref · C_k / C_total`` each cycle while the smaller side
holds; caps only ever discharge to GND, never re-charge to V_ref
during the SAR loop.  The single-ended version was retired in the move
to the differential CDAC topology; a differential Set-and-Down variant
is planned but :meth:`SarAdcMono.convert` currently raises
``NotImplementedError``.

Fabrication state (two Pelgrom-mismatched cap arrays + comparator
offset) is fully wired here — only the convert kernel awaits
implementation.  Use :class:`neurox.analog.adc.McsSarAdc` for any
current work that needs a differential SAR.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.nonideality import (
    apply_gaussian,
    apply_pelgrom_mismatch,
)

from .base import ADC


@dataclass(frozen=True)
class SarAdcMonoConfig:
    """Immutable design-parameter config for :class:`SarAdcMono`.

    Attributes:
        max_bits: Physical bit width.  Number of CDAC stages laid out
            in silicon.  The MSB cap is dropped; the active cap array
            has ``max_bits - 1`` binary-weighted caps + a dummy cap.
        v_refs: Supported reference voltages, in input units.
            ``v_refs[0]`` is the maximum (the calibration anchor).
        clk_period__ns: SAR comparator clock period.  Latency at
            ``bits`` active bits is ``(bits + 1) · clk_period``.
        c_unit__fF: Unit capacitance of the binary-weighted CDAC.
        cap_mismatch_sigma_relative: Per-unit-cap relative-σ Pelgrom
            sigma (eg. ``0.01`` for 1% matching).  ``None`` = ideal
            CDAC.
        comparator_offset: Static Gaussian config on the comparator
            threshold (sigma in [V]).  ``None`` = no static offset.
        comparator_noise: Dynamic per-cycle Gaussian comparator noise
            (sigma in [V]).  ``None`` = no comparator noise.
        kt_c_noise_enabled: When ``True``, sampling adds a Gaussian
            of σ = sqrt(k_B · T / C_total) to the held top plates.
        e_bootstrap__fJ: Constant per-conversion energy charged to
            the bootstrapped sampling switches.  ``0.0`` to disable.
        e_compare_per_bit__fJ: Energy per comparator decision
            (charged on every cycle including the MSB).
        e_logic_per_bit__fJ: Energy per cycle attributed to the SAR
            digital logic / register / control path.
        leakage_per_inst__uW: Static leakage per ADC instance.
        area_per_inst__um2: Silicon area per ADC instance.

    Note:
        Operating temperature is intentionally **not** a config field
        — it is a changeable operating-state quantity passed to the
        ADC class as an init arg.
    """

    max_bits: int
    v_refs: tuple[float, ...]

    clk_period__ns: float
    c_unit__fF: float

    cap_mismatch_sigma_relative: float | None
    comparator_offset: float | None
    comparator_noise: float | None

    kt_c_noise_enabled: bool

    e_bootstrap__fJ: float
    e_compare_per_bit__fJ: float
    e_logic_per_bit__fJ: float

    leakage_per_inst__uW: float
    area_per_inst__um2: float

    def __post_init__(self) -> None:
        if self.max_bits < 2:
            raise ValueError(
                f"SarAdcMonoConfig.max_bits ({self.max_bits}) must be >= 2 "
                f"(MSB-free design needs at least one active cap)"
            )
        if not self.v_refs:
            raise ValueError("SarAdcMonoConfig.v_refs must contain at least one V_ref")
        for i, v in enumerate(self.v_refs):
            if not (v > 0.0):
                raise ValueError(f"SarAdcMonoConfig.v_refs[{i}] ({v}) must be > 0")
        if self.clk_period__ns <= 0:
            raise ValueError(f"SarAdcMonoConfig.clk_period__ns ({self.clk_period__ns}) must be > 0")
        if self.c_unit__fF <= 0:
            raise ValueError(f"SarAdcMonoConfig.c_unit__fF ({self.c_unit__fF}) must be > 0")
        if self.e_bootstrap__fJ < 0:
            raise ValueError(f"SarAdcMonoConfig.e_bootstrap__fJ ({self.e_bootstrap__fJ}) must be >= 0")
        if self.e_compare_per_bit__fJ < 0:
            raise ValueError(f"SarAdcMonoConfig.e_compare_per_bit__fJ ({self.e_compare_per_bit__fJ}) must be >= 0")
        if self.e_logic_per_bit__fJ < 0:
            raise ValueError(f"SarAdcMonoConfig.e_logic_per_bit__fJ ({self.e_logic_per_bit__fJ}) must be >= 0")


class SarAdcMono(ADC):
    """Monotonic (Set-and-Down) differential SAR ADC — placeholder.

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

    nominal_cap_weights__fF: Tensor
    nominal_comparator_offset__V: Tensor
    c_p__fF: Tensor
    c_n__fF: Tensor
    comparator_offset__V: Tensor

    def __init__(
        self,
        cfg: SarAdcMonoConfig,
        *,
        name: str = "",
        T__K: float,
        stochastic: bool | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(name=name)
        if not (T__K > 0.0):
            raise ValueError(f"SarAdcMono T__K ({T__K}) must be > 0")
        self.cfg = cfg
        self.dtype = dtype
        self.stochastic: bool | None = stochastic
        self.T__K: float = T__K

        n_caps = cfg.max_bits - 1
        nominal_cap_weights__fF = torch.tensor(
            [cfg.c_unit__fF * (2**k) for k in range(n_caps)],
            dtype=dtype,
        )
        # Nominal templates — built once at ``__init__``, never overwritten.
        self.register_buffer("nominal_cap_weights__fF", nominal_cap_weights__fF, persistent=False)
        self.register_buffer(
            "nominal_comparator_offset__V",
            torch.zeros((), dtype=dtype),
            persistent=False,
        )
        # Sentinel fabricated buffers — :meth:`fabricate` overwrites.
        # Initialised to fresh clones of the nominals (no expand) so
        # ``.to(device)`` migrates cleanly even pre-fabricate.
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

    # --- runtime-mode introspection --- #

    def available_modes(self) -> tuple[float, ...]:
        """V_ref values the configured CDAC supports, in index order."""
        return self.cfg.v_refs

    def max_bits(self) -> int:
        """Physical CDAC bit width — the maximum active ``bits`` value."""
        return self.cfg.max_bits

    # --- fabricate (static non-idealities) --- #

    def fabricate(self, shape: tuple[int, ...]) -> None:
        """Sample the static per-instance non-idealities and store them.

        Each fabricated buffer is rebuilt from its nominal template
        via ``clone().expand(...)`` so re-calling :meth:`fabricate`
        always restarts from the unchanged nominal.  When the
        corresponding sigma is ``None`` the fabricated buffer stays a
        1-element view (no full-shape memory).

        Args:
            shape: Per-instance prefix shape.  ``()`` registers a
                single shared instance state (one ADC).  A larger
                shape registers per-instance independent fabrications.
        """
        cfg = self.cfg
        n_caps = cfg.max_bits - 1

        c_p__fF = self.nominal_cap_weights__fF.clone().expand(*shape, n_caps)
        c_p__fF = apply_pelgrom_mismatch(
            c_p__fF,
            cfg.cap_mismatch_sigma_relative,
            unit=cfg.c_unit__fF,
            floor=0.1 * cfg.c_unit__fF,
        )
        c_n__fF = self.nominal_cap_weights__fF.clone().expand(*shape, n_caps)
        c_n__fF = apply_pelgrom_mismatch(
            c_n__fF,
            cfg.cap_mismatch_sigma_relative,
            unit=cfg.c_unit__fF,
            floor=0.1 * cfg.c_unit__fF,
        )

        comparator_offset__V = self.nominal_comparator_offset__V.clone().expand(shape)
        if cfg.comparator_offset is not None:
            comparator_offset__V = apply_gaussian(comparator_offset__V, cfg.comparator_offset)

        self.register_buffer("c_p__fF", c_p__fF, persistent=False)
        self.register_buffer("c_n__fF", c_n__fF, persistent=False)
        self.register_buffer("comparator_offset__V", comparator_offset__V, persistent=False)

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

        Args:
            bits: Active bit width — required.  ``1 ≤ bits ≤ max_bits``.
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
        """Differential monotonic SAR conversion — not yet implemented."""
        del v_pos__V, v_neg__V, mode, bits
        raise NotImplementedError("Differential monotonic SAR is not yet implemented; use McsSarAdc.")

    # --- shared helpers --- #

    def _validate_runtime_args(self, mode: int, bits: int) -> None:
        """Validate per-call ``(mode, bits)``."""
        cfg = self.cfg
        if not (0 <= mode < len(cfg.v_refs)):
            raise ValueError(f"mode {mode} outside [0, {len(cfg.v_refs)})")
        if not (1 <= bits <= cfg.max_bits):
            raise ValueError(f"bits {bits} outside [1, {cfg.max_bits}]")
