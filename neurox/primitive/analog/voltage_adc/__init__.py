"""Voltage-domain ADC family — abstract base plus concrete implementations.

See also:
    docs/reference/primitive/analog/voltage_adc/README.md
"""

from neurox.primitive.analog.adc_common import AdcCalibrationRecord, AdcMode, AdcOperationPoint

from .base import VoltageAdc, VoltageAdcConfig, VoltageAdcObservation, VoltageAdcPolicy, VoltageAdcProber
from .general import GeneralVoltageAdc, GeneralVoltageAdcConfig, GeneralVoltageAdcPolicy
from .mcs_sar import McsSarVoltageAdc, McsSarVoltageAdcConfig, McsSarVoltageAdcPolicy
from .sar_mono import SarMonoVoltageAdc, SarMonoVoltageAdcConfig, SarMonoVoltageAdcPolicy

__all__ = [
    "AdcCalibrationRecord",
    "AdcMode",
    "AdcOperationPoint",
    "GeneralVoltageAdc",
    "GeneralVoltageAdcConfig",
    "GeneralVoltageAdcPolicy",
    "McsSarVoltageAdc",
    "McsSarVoltageAdcConfig",
    "McsSarVoltageAdcPolicy",
    "SarMonoVoltageAdc",
    "SarMonoVoltageAdcConfig",
    "SarMonoVoltageAdcPolicy",
    "VoltageAdc",
    "VoltageAdcConfig",
    "VoltageAdcObservation",
    "VoltageAdcPolicy",
    "VoltageAdcProber",
]
