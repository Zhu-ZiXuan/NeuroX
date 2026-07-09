"""Shared logging setup for offline tool CLIs.

See also:
    docs/guides/calibration/README.md
"""

from __future__ import annotations

import logging


def config_tool_logging(level: int = logging.INFO) -> None:
    """Configure root logging for an offline tool CLI.

    Args:
        level: Minimum level to emit. Defaults to :data:`logging.INFO`.
    """
    logging.basicConfig(level=level, format="%(message)s")
