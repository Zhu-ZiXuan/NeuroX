"""Public API for constructing and running NeuroX compute units."""

from neurox.architecture.unit import (
    Conv2dUnit,
    IdealConv2dUnit,
    IdealConv2dUnitConfig,
    IdealConv2dUnitPolicy,
    IdealLinearUnit,
    IdealLinearUnitConfig,
    IdealLinearUnitPolicy,
    LinearUnit,
    UnitBase,
)
from neurox.architecture.unit.cim import (
    CimUnit,
    CimUnitConfig,
    CimUnitPolicy,
    Conv2dCimUnit,
    Conv2dCimUnitConfig,
    Conv2dCimUnitPolicy,
    LinearCimUnit,
    LinearCimUnitConfig,
    LinearCimUnitPolicy,
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
    # unit operator surface
    "Conv2dUnit",
    "IdealConv2dUnit",
    "IdealConv2dUnitConfig",
    "IdealConv2dUnitPolicy",
    "IdealLinearUnit",
    "IdealLinearUnitConfig",
    "IdealLinearUnitPolicy",
    "LinearUnit",
    "UnitBase",
    # compute-unit public surface
    "CimUnit",
    "CimUnitConfig",
    "CimUnitPolicy",
    "Conv2dCimUnit",
    "Conv2dCimUnitConfig",
    "Conv2dCimUnitPolicy",
    "LinearCimUnit",
    "LinearCimUnitConfig",
    "LinearCimUnitPolicy",
]
