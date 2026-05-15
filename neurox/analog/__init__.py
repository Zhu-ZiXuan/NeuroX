"""Analog primitives.

The detailed ADC and readout families are intentionally **not**
re-exported here.  Callers reach the ADC subclasses through
``from neurox.analog.adc import ...`` and the readout subclasses
through ``from neurox.analog.readout import ...`` — each subpackage
owns its own public surface so the top-level ``neurox.analog``
namespace stays small (OpAmpTIA, AnalogMux, SwitchCap, Driver, DAC,
Decoder, ClampDriver and their configs).

Factory aliases (``OpAmpTIAFactory``, etc.) are intentionally **not**
provided — callers write the factory type inline as
``Callable[[], OpAmpTIA]`` at the annotation site.
"""

from .analog_mux import AnalogMux, AnalogMuxConfig
from .clamp_driver import ClampDriver
from .dac import DAC, GeneralDAC, GeneralDACConfig
from .decoder import Decoder, DecoderConfig
from .driver import Driver, DriverConfig, DriverDC, DriverSnapshot
from .opamp_tia import OpAmpTIA, OpAmpTIAConfig, OpAmpTIADC, OpAmpTIASnapshot
from .switch_cap import SwitchCap, SwitchCapConfig

__all__ = [
    "AnalogMux",
    "AnalogMuxConfig",
    "ClampDriver",
    "DAC",
    "Decoder",
    "DecoderConfig",
    "Driver",
    "DriverConfig",
    "DriverDC",
    "DriverSnapshot",
    "GeneralDAC",
    "GeneralDACConfig",
    "OpAmpTIA",
    "OpAmpTIAConfig",
    "OpAmpTIADC",
    "OpAmpTIASnapshot",
    "SwitchCap",
    "SwitchCapConfig",
]
