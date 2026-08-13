"""Ye 2023 JSSC WH-2T1R RRAM CIM macro scheme.

Importing the package registers the macro and its lookup cell with their family
registries, so `from_config` dispatches to them.
"""

from .macro import Ye2023JsscCimMacro, Ye2023JsscCimMacroConfig, Ye2023JsscCimMacroPolicy

__all__ = [
    "Ye2023JsscCimMacro",
    "Ye2023JsscCimMacroConfig",
    "Ye2023JsscCimMacroPolicy",
]
