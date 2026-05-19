"""ADC family — abstract base plus concrete implementations.

See also:
    docs/dev/modules/analog/adc/README.md
"""

from .base import ADC, ADCConfig, ADCMode
from .general import GeneralADC, GeneralADCConfig
from .mcs_sar import McsSarAdc, McsSarAdcConfig
from .sar_mono import SarAdcMono, SarAdcMonoConfig

__all__ = [
    "ADC",
    "ADCConfig",
    "ADCMode",
    "GeneralADC",
    "GeneralADCConfig",
    "McsSarAdc",
    "McsSarAdcConfig",
    "SarAdcMono",
    "SarAdcMonoConfig",
]
