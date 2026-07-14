"""Top-level NeuroX API."""

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
