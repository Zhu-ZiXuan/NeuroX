"""Runtime and artifact arguments shared by offline commands."""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ["add_runtime_args", "add_plot_args", "positive_int"]

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def positive_int(value: str) -> int:
    """Parse a positive command-line integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def add_runtime_args(parser: argparse.ArgumentParser, *, output_dir: Path, device: bool = True) -> None:
    """Add shared runtime controls; `output_dir` is the parent of each run.

    `--log-dir` is an alias of `--output-dir`. A command that performs no
    device computation opts out of the required device argument.
    """
    if device:
        parser.add_argument("--device", required=True, help="Torch device, such as cpu or cuda:0.")
    parser.add_argument(
        "--output-dir",
        "--log-dir",
        dest="output_dir",
        type=Path,
        default=output_dir,
        help="Parent directory for independent timestamped runs.",
    )
    parser.add_argument("--log-level", type=str.upper, default="INFO", choices=_LOG_LEVELS)


def add_plot_args(parser: argparse.ArgumentParser) -> None:
    """Add a plot-directory override to a command that emits figures."""
    parser.add_argument("--plot-dir", type=Path, default=None, help="Plot directory; defaults to this run's figures/.")
