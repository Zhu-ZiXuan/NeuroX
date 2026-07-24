"""Differential voltage-domain ADC family — abstract base plus concrete implementations.

See also:
    docs/reference/primitive/analog/voltage_adc/README.md
"""

from .base import (
    DifferentialVoltageAdc,
    DifferentialVoltageAdcConfig,
    DifferentialVoltageAdcObservation,
    DifferentialVoltageAdcPolicy,
    DifferentialVoltageAdcProber,
)
from .general import (
    GeneralDifferentialVoltageAdc,
    GeneralDifferentialVoltageAdcConfig,
    GeneralDifferentialVoltageAdcPolicy,
)
from .mcs_sar import (
    McsSarDifferentialVoltageAdc,
    McsSarDifferentialVoltageAdcConfig,
    McsSarDifferentialVoltageAdcPolicy,
)

__all__ = [
    "DifferentialVoltageAdc",
    "DifferentialVoltageAdcConfig",
    "DifferentialVoltageAdcObservation",
    "DifferentialVoltageAdcPolicy",
    "DifferentialVoltageAdcProber",
    "GeneralDifferentialVoltageAdc",
    "GeneralDifferentialVoltageAdcConfig",
    "GeneralDifferentialVoltageAdcPolicy",
    "McsSarDifferentialVoltageAdc",
    "McsSarDifferentialVoltageAdcConfig",
    "McsSarDifferentialVoltageAdcPolicy",
]
