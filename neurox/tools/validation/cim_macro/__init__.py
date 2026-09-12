"""Reusable construction and measurement capabilities for CIM-macro validation."""

from .cli import ValidationArgs, get_cli_args, parse_cli_args, validation_run
from .construction import build_macro
from .ppa import (
    VmmProfile,
    area_per_macro__um2,
    mean_by_name,
    profile_vmm,
    render_run_summary,
    static_energy_by_name__fJ,
)

__all__ = [
    "ValidationArgs",
    "VmmProfile",
    "area_per_macro__um2",
    "build_macro",
    "validation_run",
    "get_cli_args",
    "mean_by_name",
    "parse_cli_args",
    "profile_vmm",
    "render_run_summary",
    "static_energy_by_name__fJ",
]
