"""Scoped message-only logging for offline commands."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = ["run_logging"]


@contextmanager
def run_logging(log_path: Path, *, level: str) -> Iterator[None]:
    """Route root logging to stderr and a run file for the duration of a command.

    Each line uses the message-only format. This scope owns the root logger
    exclusively while active and restores its previous handlers and level
    on exit. Owned handlers are closed on both success and failure.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    console_handler = logging.StreamHandler()
    handlers: list[logging.Handler] = [console_handler, file_handler]
    formatter = logging.Formatter("%(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
    root = logging.getLogger()
    previous_handlers, previous_level = root.handlers[:], root.level
    try:
        root.handlers[:] = handlers
        root.setLevel(level.upper())
        yield
    finally:
        root.handlers[:] = previous_handlers
        root.setLevel(previous_level)
        for handler in handlers:
            handler.close()
