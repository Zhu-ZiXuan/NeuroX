"""TIA family — abstract base plus concrete implementations.

See also:
    docs/reference/primitive/analog/tia/README.md
"""

from .base import Tia, TiaConfig, TiaPolicy, TiaSnap
from .general import GeneralTia, GeneralTiaConfig, GeneralTiaDcop, GeneralTiaPolicy, GeneralTiaSnap
from .opamp_tia import OpAmpTia, OpAmpTiaConfig, OpAmpTiaDcop, OpAmpTiaPolicy, OpAmpTiaSnap

__all__ = [
    "GeneralTia",
    "GeneralTiaConfig",
    "GeneralTiaDcop",
    "GeneralTiaPolicy",
    "GeneralTiaSnap",
    "OpAmpTia",
    "OpAmpTiaConfig",
    "OpAmpTiaDcop",
    "OpAmpTiaPolicy",
    "OpAmpTiaSnap",
    "Tia",
    "TiaConfig",
    "TiaPolicy",
    "TiaSnap",
]
