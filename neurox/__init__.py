"""Top-level NeuroX API.

Public surface stops at the architecture layer. Quantization operators, training
pipelines, and graph-rewrite tooling live in application code above this
surface (each application owns its own pipeline; nothing shared). Pure quant
primitives that any pipeline might compose (stochastic rounding, observers,
fake-quant, fixed-point scale conversion) live in ``neurox.common.quant``.
"""

from neurox.common.mixin import ProfileMixin
from neurox.common.profiler import (
    EnergyEvent,
    LatencyEvent,
    NeuroxProfiler,
    ProfilerReport,
    StaticMetrics,
    StaticRecord,
)
from neurox.primitive.macro.cim import (
    CimMacro,
    CimMacroConfig,
    CimMacroPolicy,
    IdealCimMacro,
    IdealCimMacroConfig,
    IdealCimMacroPolicy,
)
from neurox.architecture.unit import QuantMatMul
from neurox.architecture.unit.cim import (
    CimUnit,
    CimUnitConfig,
    CimUnitPolicy,
    DirectCimUnit,
    DirectCimUnitConfig,
    DirectCimUnitPolicy,
    IdealCimUnit,
    IdealCimUnitConfig,
    IdealCimUnitPolicy,
    InterArraySliceCimUnit,
    InterArraySliceCimUnitConfig,
    InterArraySliceCimUnitPolicy,
    IntraArraySliceCimUnit,
    IntraArraySliceCimUnitConfig,
    IntraArraySliceCimUnitPolicy,
)

__all__ = [
    # profiler
    "EnergyEvent",
    "LatencyEvent",
    "NeuroxProfiler",
    "ProfileMixin",
    "ProfilerReport",
    "StaticMetrics",
    "StaticRecord",
    # macro tile public surface
    "CimMacro",
    "CimMacroConfig",
    "CimMacroPolicy",
    "IdealCimMacro",
    "IdealCimMacroConfig",
    "IdealCimMacroPolicy",
    # compute-unit public surface
    "CimUnit",
    "CimUnitConfig",
    "CimUnitPolicy",
    "DirectCimUnit",
    "DirectCimUnitConfig",
    "DirectCimUnitPolicy",
    "IdealCimUnit",
    "IdealCimUnitConfig",
    "IdealCimUnitPolicy",
    "InterArraySliceCimUnit",
    "InterArraySliceCimUnitConfig",
    "InterArraySliceCimUnitPolicy",
    "IntraArraySliceCimUnit",
    "IntraArraySliceCimUnitConfig",
    "IntraArraySliceCimUnitPolicy",
    "QuantMatMul",
]
