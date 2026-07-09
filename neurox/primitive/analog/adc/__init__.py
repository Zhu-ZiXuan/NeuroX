"""ADC family — abstract base plus concrete implementations.

See also:
    docs/reference/primitive/analog/adc/README.md
"""

from .base import ADC, AdcCalibrationRecord, ADCConfig, ADCMode, AdcOperationPoint, ADCPolicy
from .general import GeneralADC, GeneralADCConfig, GeneralADCPolicy
from .mcs_sar import McsSarAdc, McsSarAdcConfig, McsSarAdcPolicy

__all__ = [
    "ADC",
    "ADCConfig",
    "ADCMode",
    "ADCPolicy",
    "AdcOperationPoint",
    "AdcCalibrationRecord",
    "GeneralADC",
    "GeneralADCConfig",
    "GeneralADCPolicy",
    "McsSarAdc",
    "McsSarAdcConfig",
    "McsSarAdcPolicy",
]
