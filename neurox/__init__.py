"""Top-level NeuroX API.

Public surface stops at the macro layer. Quantization operators, training
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
from neurox.macro import NeuroxMacroQuantMatMul
from neurox.macro.xbar import (
    DirectXbarMacro,
    DirectXbarMacroConfig,
    DirectXbarMacroPolicy,
    IdealXbarMacro,
    IdealXbarMacroConfig,
    IdealXbarMacroPolicy,
    InterArraySliceXbarMacro,
    InterArraySliceXbarMacroConfig,
    InterArraySliceXbarMacroPolicy,
    IntraArraySliceXbarMacro,
    IntraArraySliceXbarMacroConfig,
    IntraArraySliceXbarMacroPolicy,
    XbarMacro,
    XbarMacroConfig,
    XbarMacroPolicy,
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
    # macro public surface
    "DirectXbarMacro",
    "DirectXbarMacroConfig",
    "DirectXbarMacroPolicy",
    "IdealXbarMacro",
    "IdealXbarMacroConfig",
    "IdealXbarMacroPolicy",
    "InterArraySliceXbarMacro",
    "InterArraySliceXbarMacroConfig",
    "InterArraySliceXbarMacroPolicy",
    "IntraArraySliceXbarMacro",
    "IntraArraySliceXbarMacroConfig",
    "IntraArraySliceXbarMacroPolicy",
    "NeuroxMacroQuantMatMul",
    "XbarMacro",
    "XbarMacroConfig",
    "XbarMacroPolicy",
]
