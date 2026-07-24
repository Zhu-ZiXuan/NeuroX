"""Single-ended current-domain ADC family — abstract base plus concrete implementations.

See also:
    docs/reference/primitive/analog/current_adc/README.md
"""

from .base import (
    SingleEndedCurrentAdc,
    SingleEndedCurrentAdcConfig,
    SingleEndedCurrentAdcObservation,
    SingleEndedCurrentAdcPolicy,
    SingleEndedCurrentAdcProber,
)
from .sar import SarSingleEndedCurrentAdc, SarSingleEndedCurrentAdcConfig, SarSingleEndedCurrentAdcPolicy

__all__ = [
    "SarSingleEndedCurrentAdc",
    "SarSingleEndedCurrentAdcConfig",
    "SarSingleEndedCurrentAdcPolicy",
    "SingleEndedCurrentAdc",
    "SingleEndedCurrentAdcConfig",
    "SingleEndedCurrentAdcObservation",
    "SingleEndedCurrentAdcPolicy",
    "SingleEndedCurrentAdcProber",
]
