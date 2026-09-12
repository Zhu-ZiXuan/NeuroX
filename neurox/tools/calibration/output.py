"""Reporting and persistence of calibration configuration fragments."""

from __future__ import annotations

import logging
from pathlib import Path

from neurox.tools.output import RunOutput, atomic_output

__all__ = ["emit_fragment"]

logger = logging.getLogger(__name__)


def emit_fragment(text: str, output: RunOutput, *, name: str, destination: Path | None = None) -> Path:
    """Log a TOML fragment and retain it in the run, with an optional extra copy."""
    text = text.rstrip() + "\n"
    logger.info("%s", text.rstrip())
    path = output.write_text(name, text)
    if destination is not None and destination.resolve() != path.resolve():
        with atomic_output(destination) as temporary:
            temporary.write_text(text, encoding="utf-8")
        logger.info("wrote TOML fragment to %s", destination)
    logger.info("run fragment: %s", path)
    return path
