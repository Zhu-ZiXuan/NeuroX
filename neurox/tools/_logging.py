"""Shared logging setup for offline tool CLIs."""

from __future__ import annotations

import logging


def config_tool_logging(level: int = logging.INFO) -> None:
    """Configure root logging for an offline tool CLI."""
    logging.basicConfig(level=level, format="%(message)s")
