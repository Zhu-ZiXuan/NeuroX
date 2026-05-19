"""Default configuration files bundled with NeuroX."""

from pathlib import Path

CONFIG_DIR: Path = Path(__file__).parent

DEFAULT_1T1R_TOML: Path = CONFIG_DIR / "default_1t1r.toml"

__all__ = [
    "CONFIG_DIR",
    "DEFAULT_1T1R_TOML",
]
