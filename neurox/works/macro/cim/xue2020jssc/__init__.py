"""Xue 2020 JSSC RRAM CIM macro scheme — one sub-array of the published 1-Mb macro.

Importing the package registers the macro with the CIM-macro family registry, so
`from_config` dispatches to it; the scheme's sub-blocks are owner-built concrete
classes with no registry entries.

@article{xue2020jssc,
    author  = {TODO: only "Xue et al." was salvageable},
    title   = {Embedded 1-Mb ReRAM-Based Computing-in-Memory Macro With Multibit Input
               and Weight for CNN-Based AI Edge Processors},
    journal = {IEEE Journal of Solid-State Circuits},
    year    = {2020},
    volume  = {TODO},
    number  = {TODO},
    pages   = {TODO},
    doi     = {10.1109/JSSC.2019.2951363},
}
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
