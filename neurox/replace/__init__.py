"""Inference-side replacement pipeline."""

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
    "NeuroxStateError",
    "ReplacementContext",
    "ReplacementPolicy",
    "ReplacementRule",
    "StateBindingReport",
    "StructuralReport",
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
    "replace_model",
    "report_replacement",
]
