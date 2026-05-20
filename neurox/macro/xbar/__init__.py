"""Xbar-backed macros.

See also:
    docs/dev/architecture/xbar_macro.md
"""

from .base import XbarMacro, XbarMacroConfig
from .inter_xbar_slice import InterXbarSliceMacro, InterXbarSliceMacroConfig
from .intra_xbar_slice import IntraXbarSliceMacro, IntraXbarSliceMacroConfig

__all__ = [
    "InterXbarSliceMacro",
    "InterXbarSliceMacroConfig",
    "IntraXbarSliceMacro",
    "IntraXbarSliceMacroConfig",
    "XbarMacro",
    "XbarMacroConfig",
]
