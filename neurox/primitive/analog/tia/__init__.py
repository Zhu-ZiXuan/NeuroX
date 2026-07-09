"""TIA family — abstract base plus concrete implementations.

See also:
    docs/reference/primitive/analog/tia/README.md
"""

from .base import TIA, TIAConfig, TIAPolicy, TIASnap
from .general import GeneralTIA, GeneralTIAConfig, GeneralTIADCOP, GeneralTIAPolicy, GeneralTIASnap
from .opamp_tia import OpAmpTIA, OpAmpTIAConfig, OpAmpTIADCOP, OpAmpTIAPolicy, OpAmpTIASnap

__all__ = [
    "GeneralTIA",
    "GeneralTIAConfig",
    "GeneralTIADCOP",
    "GeneralTIAPolicy",
    "GeneralTIASnap",
    "OpAmpTIA",
    "OpAmpTIAConfig",
    "OpAmpTIADCOP",
    "OpAmpTIAPolicy",
    "OpAmpTIASnap",
    "TIA",
    "TIAConfig",
    "TIAPolicy",
    "TIASnap",
]
