"""Shared logging setup for offline tool CLIs."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path


def config_tool_logging(level: int = logging.INFO) -> None:
    """Configure root logging for an offline tool CLI."""
    logging.basicConfig(level=level, format="%(message)s")


def create_run_directory(output_dir: Path, tool_name: str) -> Path:
    """Create one timestamped child directory for an offline tool run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{tool_name}_{stamp}"
    for index in range(1000):
        suffix = "" if index == 0 else f"_{index + 1}"
        run_dir = output_dir / f"{base_name}{suffix}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir
    raise FileExistsError(f"could not allocate a run directory under {output_dir}")


def attach_file_logging(log_path: Path) -> Path:
    """Attach one plain-message file handler at an exact path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)
    return log_path


def add_file_logging(log_dir: Path, tool_name: str) -> Path:
    """Attach one plain-message file handler and return its timestamped path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{tool_name}_{stamp}.log"
    return attach_file_logging(log_path)
