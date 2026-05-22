"""Xbar-backed macros.

See also:
    docs/dev/architecture/xbar_macro.md
"""

from .base import XbarMacro, XbarMacroConfig
from .direct import DirectXbarMacro, DirectXbarMacroConfig
from .inter_array_slice import InterArraySliceXbarMacro, InterArraySliceXbarMacroConfig
from .intra_array_slice import IntraArraySliceXbarMacro, IntraArraySliceXbarMacroConfig

__all__ = [
    "DirectXbarMacro",
    "DirectXbarMacroConfig",
    "InterArraySliceXbarMacro",
    "InterArraySliceXbarMacroConfig",
    "IntraArraySliceXbarMacro",
    "IntraArraySliceXbarMacroConfig",
    "XbarMacro",
    "XbarMacroConfig",
]
