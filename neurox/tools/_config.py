"""Shared helpers for the config-driven CLI tools under :mod:`neurox.tools`.

All tools follow the same pattern: a TOML config carries every input that
affects the result (chip preset, workload, sweep ranges, numerical knobs,
seed, dtype), while CLI keeps only runtime / output knobs (target device,
output paths, log level).

This module centralises the boilerplate that pattern requires.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TypeVar

from neurox.common import dataclass_from_file
from neurox.tools.logging import config_tool_logging

_T = TypeVar("_T")


_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def add_standard_args(
    parser: argparse.ArgumentParser,
    *,
    device: bool = True,
    plot_dir: bool = False,
    plot_file: bool = False,
    output_file: bool = False,
) -> None:
    """Append the standard runtime / output flags to ``parser``.

    All tools take ``--config`` and ``--log-level``. ``--device`` is
    opt-in (default ``True``): pass ``device=False`` for tools that run
    in fixed precision on CPU so the CLI surface does not expose a knob
    the tool would silently ignore. Output flags are opt-in via keyword
    arguments.

    Args:
        parser: The argparse parser to add flags to.
        device: Whether to add ``--device`` (default ``True``).
        plot_dir: Add a ``--plot-dir`` flag for tools that emit multiple
            PNGs into one directory.
        plot_file: Add a ``--plot`` flag for tools that emit a single PNG.
        output_file: Add a ``--output`` flag for tools that emit a TOML.
    """
    parser.add_argument("--config", type=Path, required=True, help="Tool-run TOML config path")
    if device:
        parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            help="Torch device (e.g. 'cuda:0', 'cpu'). Defaults to 'cpu' — no implicit GPU pickup.",
        )
    if plot_dir:
        parser.add_argument(
            "--plot-dir",
            type=Path,
            default=None,
            help="Optional directory for plot PNGs",
        )
    if plot_file:
        parser.add_argument(
            "--plot",
            type=Path,
            default=None,
            help="Optional output PNG path",
        )
    if output_file:
        parser.add_argument(
            "--output",
            type=Path,
            default=None,
            help="Optional output TOML path",
        )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        default="INFO",
        choices=_LOG_LEVELS,
        help=f"Logger level; one of {{{', '.join(_LOG_LEVELS)}}}",
    )


def setup_logging(level_name: str) -> None:
    """Configure root logging based on ``--log-level``."""
    config_tool_logging(level=getattr(logging, level_name.upper()))


def load_tool_config(cls: type[_T], config_path: Path) -> _T:
    """Parse the tool-run TOML into ``cls`` via :func:`dataclass_from_file`.

    Convenience over calling ``dataclass_from_file`` directly: the
    function exists so tool-side imports stay shallow.
    """
    return dataclass_from_file(cls, config_path)


def resolve_relative_path(path: Path | str | None, base: Path) -> Path | None:
    """Resolve a TOML-supplied path against ``base``'s directory.

    ``None`` propagates as ``None`` so optional path fields stay
    declarative. Absolute paths are returned unchanged.

    Args:
        path: Raw value from the TOML (may be ``None``, ``str``, or ``Path``).
        base: The TOML file's path; paths resolve relative to ``base.parent``.
    """
    if path is None:
        return None
    p = Path(path) if not isinstance(path, Path) else path
    if p.is_absolute():
        return p
    return base.parent / p
