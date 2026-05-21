"""Default configuration files bundled with NeuroX."""

from pathlib import Path

CONFIG_DIR: Path = Path(__file__).parent

DEFAULT_1T1R_MACRO_TOML: Path = CONFIG_DIR / "default_1t1r_macro.toml"
DEFAULT_1T1R_XBAR_TOML: Path = CONFIG_DIR / "default_1t1r_xbar.toml"

__all__ = [
    "CONFIG_DIR",
    "DEFAULT_1T1R_MACRO_TOML",
    "DEFAULT_1T1R_XBAR_TOML",
]
