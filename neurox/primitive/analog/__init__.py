"""NeuroX analog primitives.

See also:
    docs/reference/primitive/analog/README.md
"""

from .adc_common import AdcCalibrationRecord, AdcMode, AdcOperationPoint
from .current_adc import (
    CurrentAdc,
    CurrentAdcConfig,
    CurrentAdcPolicy,
    SarCurrentAdc,
    SarCurrentAdcConfig,
    SarCurrentAdcPolicy,
)
from .current_dac import (
    CurrentDac,
    CurrentDacConfig,
    CurrentDacPolicy,
    GeneralCurrentDac,
    GeneralCurrentDacConfig,
    GeneralCurrentDacPolicy,
)
from .current_mirror import CurrentMirror, CurrentMirrorConfig, CurrentMirrorPolicy
from .current_mux import CurrentMux, CurrentMuxConfig, CurrentMuxPolicy
from .current_reference import (
    CurrentReference,
    CurrentReferenceConfig,
    CurrentReferencePolicy,
    CurrentReferenceSnap,
)
from .current_subtractor import (
    CurrentSubtractor,
    CurrentSubtractorConfig,
    CurrentSubtractorPolicy,
)
from .switch_cap import SwitchCap, SwitchCapConfig, SwitchCapPolicy
from .voltage_adc import (
    GeneralVoltageAdc,
    GeneralVoltageAdcConfig,
    GeneralVoltageAdcPolicy,
    McsSarVoltageAdc,
    McsSarVoltageAdcConfig,
    McsSarVoltageAdcPolicy,
    SarMonoVoltageAdc,
    SarMonoVoltageAdcConfig,
    SarMonoVoltageAdcPolicy,
    VoltageAdc,
    VoltageAdcConfig,
    VoltageAdcPolicy,
)
from .voltage_dac import (
    GeneralVoltageDac,
    GeneralVoltageDacConfig,
    GeneralVoltageDacPolicy,
    VoltageDac,
    VoltageDacConfig,
    VoltageDacPolicy,
)
from .voltage_driver import (
    VoltageDriver,
    VoltageDriverConfig,
    VoltageDriverPolicy,
    VoltageDriverSnap,
)
from .voltage_mux import VoltageMux, VoltageMuxConfig, VoltageMuxPolicy
from .voltage_reference import (
    VoltageReference,
    VoltageReferenceConfig,
    VoltageReferencePolicy,
    VoltageReferenceSnap,
)

__all__ = [
    "AdcCalibrationRecord",
    "AdcMode",
    "AdcOperationPoint",
    "CurrentAdc",
    "CurrentAdcConfig",
    "CurrentAdcPolicy",
    "CurrentDac",
    "CurrentDacConfig",
    "CurrentDacPolicy",
    "CurrentMirror",
    "CurrentMirrorConfig",
    "CurrentMirrorPolicy",
    "CurrentMux",
    "CurrentMuxConfig",
    "CurrentMuxPolicy",
    "CurrentReference",
    "CurrentReferenceConfig",
    "CurrentReferencePolicy",
    "CurrentReferenceSnap",
    "CurrentSubtractor",
    "CurrentSubtractorConfig",
    "CurrentSubtractorPolicy",
    "GeneralCurrentDac",
    "GeneralCurrentDacConfig",
    "GeneralCurrentDacPolicy",
    "GeneralVoltageAdc",
    "GeneralVoltageAdcConfig",
    "GeneralVoltageAdcPolicy",
    "GeneralVoltageDac",
    "GeneralVoltageDacConfig",
    "GeneralVoltageDacPolicy",
    "McsSarVoltageAdc",
    "McsSarVoltageAdcConfig",
    "McsSarVoltageAdcPolicy",
    "SarCurrentAdc",
    "SarCurrentAdcConfig",
    "SarCurrentAdcPolicy",
    "SarMonoVoltageAdc",
    "SarMonoVoltageAdcConfig",
    "SarMonoVoltageAdcPolicy",
    "SwitchCap",
    "SwitchCapConfig",
    "SwitchCapPolicy",
    "VoltageDriver",
    "VoltageDriverConfig",
    "VoltageDriverPolicy",
    "VoltageDriverSnap",
    "VoltageAdc",
    "VoltageAdcConfig",
    "VoltageAdcPolicy",
    "VoltageDac",
    "VoltageDacConfig",
    "VoltageDacPolicy",
    "VoltageMux",
    "VoltageMuxConfig",
    "VoltageMuxPolicy",
    "VoltageReference",
    "VoltageReferenceConfig",
    "VoltageReferencePolicy",
    "VoltageReferenceSnap",
]
