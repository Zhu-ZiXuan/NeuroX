"""Xue2020 JSSC SINWP 1T1R CIM sub-array scheme.

A single 256x512 sub-array of the 1-Mb macro of Xue et al. (IEEE JSSC 2020):
3-bit sign-magnitude weights on P/N cell pairs (MSB/LSB digits), 2-bit serial
word-line inputs, and a signed-magnitude current-mode output produced by the
DSWCT digit legs, the SINWP-SC input-radix combine, the PN-ISUB subtraction, and
a TMCSA SAR current ADC. Under the zero-wire, ideal-clamp idealization the whole
readout collapses into vectorized tensor operations over the exact linearized
1T1R cell; only kernel primitives (:mod:`neurox.primitive`) are composed.

Importing this package registers :class:`Xue2020JsscCimMacro` under its config key.
"""

from neurox.works.macro.cim.xue2020jssc import macro

__all__ = ["macro"]
