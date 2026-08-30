"""Xue et al. 2020 JSSC 1T1R CIM macro.

@article{xue2020jssc,
    author  = {Xue, Cheng-Xin and Chen, Wei-Hao and Liu, Je-Syu and Li, Jia-Fang and Lin, Wei-Yu and Lin, Wei-En and
                Wang, Jing-Hong and Wei, Wei-Chen and Huang, Tsung-Yuan and Chang, Ting-Wei and
                Chang, Tung-Cheng and Kao, Hui-Yao and Chiu, Yen-Cheng and Lee, Chun-Ying and King, Ya-Chin and
                Lin, Chrong-Jung and Liu, Ren-Shuo and Hsieh, Chih-Cheng and Tang, Kea-Tiong and Chang, Meng-Fan},
    title   = {Embedded 1-Mb ReRAM-Based Computing-in-Memory Macro With Multibit Input
                and Weight for CNN-Based AI Edge Processors},
    journal = {IEEE Journal of Solid-State Circuits},
    year    = {2020},
    month   = {Jan},
    volume  = {55},
    number  = {1},
    pages   = {203--215},
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
