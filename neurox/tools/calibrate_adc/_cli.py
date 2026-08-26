"""Command-line arguments for ADC-input characterization."""

from __future__ import annotations

import argparse
from pathlib import Path

from neurox.tools._config import add_standard_args

_DEFAULT_MIN_SAMPLES_PER_IDEAL_VALUE = 65536


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe nominal ADC-input distributions for one macro mode")
    add_standard_args(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("log/calibration"),
        help="Parent directory for the timestamped run directory (default log/calibration)",
    )
    parser.add_argument(
        "--min-samples-per-ideal-value",
        type=_positive_int,
        default=_DEFAULT_MIN_SAMPLES_PER_IDEAL_VALUE,
        help=f"Minimum combined samples for every signed ideal value (default {_DEFAULT_MIN_SAMPLES_PER_IDEAL_VALUE})",
    )
    return parser.parse_args(argv)
