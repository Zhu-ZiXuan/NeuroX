"""Default configuration files bundled with NeuroX.

Exposes package-anchored paths so callers never have to hard-code
working-directory-relative strings like ``"neurox/config/xxx.toml"``.
The paths are resolved from ``__file__``, so they work regardless of
the caller's CWD or whether ``neurox`` is installed or run from a
source checkout.
"""

from pathlib import Path

CONFIG_DIR: Path = Path(__file__).parent

DEFAULT_1T1R_TOML: Path = CONFIG_DIR / "default_1t1r.toml"

__all__ = [
    "CONFIG_DIR",
    "DEFAULT_1T1R_TOML",
]
