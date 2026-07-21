"""Current-domain ADC family — abstract base plus concrete implementations.

See also:
    docs/reference/primitive/analog/current_adc/README.md
"""

from neurox.primitive.analog.adc_common import AdcCalibrationRecord, AdcMode, AdcOperationPoint

from .base import CurrentAdc, CurrentAdcConfig, CurrentAdcObservation, CurrentAdcPolicy, CurrentAdcProber
from .sar import SarCurrentAdc, SarCurrentAdcConfig, SarCurrentAdcPolicy

__all__ = [
    "AdcCalibrationRecord",
    "AdcMode",
    "AdcOperationPoint",
    "CurrentAdc",
    "CurrentAdcConfig",
    "CurrentAdcObservation",
    "CurrentAdcPolicy",
    "CurrentAdcProber",
    "SarCurrentAdc",
    "SarCurrentAdcConfig",
    "SarCurrentAdcPolicy",
]
