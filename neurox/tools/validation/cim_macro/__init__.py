"""Reusable construction and measurement capabilities for CIM-macro validation."""

from .cli import ValidationArgs, get_cli_args, parse_cli_args, validation_run
from .construction import build_macro
from .ppa import (
    area_per_macro__um2,
    dynamic_energy_by_round__fJ,
    mean_by_name,
    profile_vmm,
    render_run_summary,
    static_energy_by_round__fJ,
)

__all__ = [
    "ValidationArgs",
    "area_per_macro__um2",
    "dynamic_energy_by_round__fJ",
    "build_macro",
    "validation_run",
    "get_cli_args",
    "mean_by_name",
    "parse_cli_args",
    "profile_vmm",
    "render_run_summary",
    "static_energy_by_round__fJ",
]
