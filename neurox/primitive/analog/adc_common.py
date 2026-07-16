"""Domain-neutral ADC types shared by the voltage and current ADC families.

See also:
    docs/reference/primitive/analog/adc_common.md
"""

from __future__ import annotations

from dataclasses import dataclass

from neurox.common.mixin import ValidateMixin


@dataclass(frozen=True, slots=True)
class AdcOperationPoint:
    """ADC operating point — the runtime selection passed per call.

    Attributes:
        adc_mode: Operating-point index selecting a reference tap from the
            injected reference tensor — valid range ``[0, num_refs)``.
        adc_bits: Active bit width, ``1 <= adc_bits <= max_bits``.
    """

    adc_mode: int
    adc_bits: int


@dataclass(frozen=True)
class AdcCalibrationRecord(ValidateMixin):
    """One row of the ADC ``adc_operation_point -> rescale_factor`` lookup table.

    Attributes:
        adc_mode: Operating-point index.
        adc_bits: Active bit width.
        rescale_factor: Recovery-side multiplier; ``M_ideal ≈ code · rescale_factor``.
            Quantize is the inverse: ``code = floor(M_ideal / rescale_factor)``.
    """

    adc_mode: int
    adc_bits: int
    rescale_factor: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._require_non_neg(self.adc_mode, "adc_mode")
        self._require_non_neg(self.adc_bits, "adc_bits")


@dataclass(frozen=True)
class AdcMode(ValidateMixin):
    """One operating mode of a multi-mode ADC.

    Domain-neutral: ``max_signal`` is the full-scale input magnitude in the
    family's native analog quantity — voltage [V] for a voltage ADC, current
    [uA] for a current ADC.

    Attributes:
        n_bits: Bit width of the mode.
        n_states: Number of analog states represented by the mode.
        max_signal: Full-scale analog input magnitude.
    """

    n_bits: int
    n_states: int
    max_signal: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.n_bits < 1:
            raise ValueError(f"require: n_bits ({self.n_bits}) >= 1")
        if self.n_states < 2:
            raise ValueError(f"require: n_states ({self.n_states}) >= 2")
        if self.n_states > (1 << self.n_bits):
            raise ValueError(f"require: n_states ({self.n_states}) <= 2**n_bits ({1 << self.n_bits})")
        if not (self.max_signal > 0.0):
            raise ValueError(f"require: max_signal ({self.max_signal}) > 0")

    @property
    def n_codes(self) -> int:
        """Number of distinct output codes — ``2 ** n_bits``."""
        return 1 << self.n_bits

    @property
    def lsb(self) -> float:
        """Bin width — ``max_signal / n_codes``."""
        return self.max_signal / self.n_codes
