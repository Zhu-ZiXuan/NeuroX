"""Xue 2020 JSSC RRAM CIM macro scheme.

Importing the package registers the macro with the CIM-macro family registry, so
`from_config` dispatches to it; the scheme's sub-blocks are owner-built concrete
classes with no registry entries.
"""

from .macro import (
    Xue2020JsscCimMacro,
    Xue2020JsscCimMacroConfig,
    Xue2020JsscCimMacroPolicy,
)

__all__ = [
    "Xue2020JsscCimMacro",
    "Xue2020JsscCimMacroConfig",
    "Xue2020JsscCimMacroPolicy",
]
