"""Ye 2023 JSSC WH-2T1R RRAM CIM macro scheme — one macro of the published 28-nm chip.

Importing the package registers the macro with its family registry.

@article{ye2023jssc,
    author  = {TODO: only "Ye et al." was salvageable},
    title   = {A 28-nm RRAM Computing-in-Memory Macro Using Weighted Hybrid 2T1R Cell
               Array and Reference Subtracting Sense Amplifier for AI Edge Inference},
    journal = {IEEE Journal of Solid-State Circuits},
    year    = {2023},
    volume  = {58},
    number  = {10},
    pages   = {2839--2848},
    doi     = {TODO},
}
"""

from .macro import Ye2023JsscCimMacro, Ye2023JsscCimMacroConfig, Ye2023JsscCimMacroPolicy

__all__ = [
    "Ye2023JsscCimMacro",
    "Ye2023JsscCimMacroConfig",
    "Ye2023JsscCimMacroPolicy",
]
