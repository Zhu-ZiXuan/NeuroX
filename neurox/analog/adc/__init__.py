"""ADC family — abstract base plus concrete implementations.

See also:
    docs/dev/modules/analog/adc/README.md
"""

from .base import ADC, AdcCalibrationRecord, ADCConfig, ADCMode, AdcOperationPoint
from .general import GeneralADC, GeneralADCConfig
from .mcs_sar import McsSarAdc, McsSarAdcConfig
from .sar_mono import SarAdcMono, SarAdcMonoConfig

__all__ = [
    "ADC",
    "ADCConfig",
    "ADCMode",
    "AdcOperationPoint",
    "AdcCalibrationRecord",
    "GeneralADC",
    "GeneralADCConfig",
    "McsSarAdc",
    "McsSarAdcConfig",
    "SarAdcMono",
    "SarAdcMonoConfig",
]
