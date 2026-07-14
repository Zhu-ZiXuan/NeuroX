"""Shared helpers for the config-driven CLI tools under :mod:`neurox.tools`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TypeVar

from neurox.common.mixin import SerializeMixin
from neurox.tools._logging import config_tool_logging

_T = TypeVar("_T", bound=SerializeMixin)


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
    """Parse the tool-run TOML into ``cls`` via :meth:`SerializeMixin.from_file`."""
    return cls.from_file(config_path)


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
