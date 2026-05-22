"""Inference-side replacement pipeline."""

from neurox.profiler import (
    NeuroxProfiler,
    ProfiledModule,
    ProfilerReport,
    RuntimeEvent,
    StaticMetrics,
    StaticRecord,
)

from .policy import (
    ReplacementContext,
    ReplacementPolicy,
    ReplacementRule,
    StructuralReport,
    by_attr_match,
    default_policy,
    heterogeneous_macro_policy,
    name_excluded_match,
)
from .replace import (
    bind_output_calibration,
    build_evaluator,
    count_unreplaced_ops,
    count_xbar_layers,
    fabricate_model,
    load_neurox_state,
    program_model,
    replace_model,
    report_replacement,
)
from .state import NeuroxStateError, StateBindingReport

__all__ = [
    # --- profiler (re-exported for convenience) ---
    "NeuroxProfiler",
    "ProfiledModule",
    "ProfilerReport",
    "RuntimeEvent",
    "StaticMetrics",
    "StaticRecord",
    # --- replace pipeline (staged surface) ---
    "bind_output_calibration",
    "build_evaluator",
    "by_attr_match",
    "count_unreplaced_ops",
    "count_xbar_layers",
    "default_policy",
    "fabricate_model",
    "heterogeneous_macro_policy",
    "load_neurox_state",
    "name_excluded_match",
    "program_model",
    "NeuroxStateError",
    "ReplacementContext",
    "ReplacementPolicy",
    "ReplacementRule",
    "replace_model",
    "report_replacement",
    "StateBindingReport",
    "StructuralReport",
]
