"""Domain-neutral ADC descriptor/calibration types (AdcMode, AdcCalibrationRecord) shared by the voltage and current ADC families.

See also:
    docs/reference/primitive/analog/adc_common.md
"""

from __future__ import annotations

from dataclasses import dataclass

from neurox.common.mixin import ValidateMixin


@dataclass(frozen=True)
class AdcCalibrationRecord(ValidateMixin):
    """``(mode, bits) -> rescale_factor`` lookup row.

    Attributes:
        mode: Operating-point index.
        bits: Active bit width.
        rescale_factor: Recovery-side multiplier. Codes are raw (unsigned /
            offset-binary), so recovery is consumer-side and affine-aware:
            ``M_ideal ≈ (code − zero) · rescale_factor``, where ``zero`` is
            the emitting ADC's zero-point offset for the operating point (0
            for a genuinely single-ended magnitude ADC). The record stores
            only the linear coefficient; the ``zero`` offset comes from the
            ADC, not from this record.
    """

    mode: int
    bits: int
    rescale_factor: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._require_non_neg(self.mode, "mode")
        self._require_non_neg(self.bits, "bits")


@dataclass(frozen=True)
class AdcMode(ValidateMixin):
    """One operating mode of a multi-mode ADC.

    Domain-neutral: ``max_signal`` is the full-scale input magnitude in the
    family's native analog quantity — voltage [V] for a voltage ADC, current
    [uA] for a current ADC.

    Attributes:
        bits: Bit width of the mode.
        n_states: Number of analog states represented by the mode.
        max_signal: Full-scale analog input magnitude.
    """

    bits: int
    n_states: int
    max_signal: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.bits < 1:
            raise ValueError(f"require: bits ({self.bits}) >= 1")
        if self.n_states < 2:
            raise ValueError(f"require: n_states ({self.n_states}) >= 2")
        if self.n_states > (1 << self.bits):
            raise ValueError(f"require: n_states ({self.n_states}) <= 2**bits ({1 << self.bits})")
        if not (self.max_signal > 0.0):
            raise ValueError(f"require: max_signal ({self.max_signal}) > 0")

    @property
    def n_codes(self) -> int:
        """Number of distinct output codes — ``2 ** bits``."""
        return 1 << self.bits

    @property
    def lsb(self) -> float:
        """Bin width — ``max_signal / n_codes``."""
        return self.max_signal / self.n_codes
