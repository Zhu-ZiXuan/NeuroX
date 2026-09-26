"""Arguments shared by independently executable calibration tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from neurox.tools.cli import add_plot_args, add_runtime_args

__all__ = ["add_calibration_args"]


def add_calibration_args(parser: argparse.ArgumentParser, *, fragment: bool = False, plots: bool = False) -> None:
    """Add config and runtime arguments, plus the task's artifact overrides."""
    add_runtime_args(parser, output_dir=Path("log/calibration"))
    parser.add_argument("--config", type=Path, required=True, help="Tool-run TOML config path.")
    if fragment:
        parser.add_argument(
            "--output", type=Path, default=None, help="Additional destination for the emitted TOML fragment."
        )
    if plots:
        add_plot_args(parser)
