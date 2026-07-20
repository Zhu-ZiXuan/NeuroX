"""Current-subtractor / current-ADC 1T1R CIM macro scheme.

A single-tile ReRAM CIM macro: ternary weights {-1, 0, +1} on P/N physical
column pairs, 1-bit inputs on the word lines, and a signed-magnitude
current-mode output (sign bit + ``n_bits`` ADC magnitude, config-driven)
produced by a P-N current subtractor feeding a SAR current ADC. Input
sub-phases, sample-and-hold capacitors, and MSB/LSB weight digits are
intentionally absent. Only kernel primitives (:mod:`neurox.primitive`) are
composed — the pure array is the kernel
:class:`~neurox.primitive.xbar.array.XbarArray1t1r`. The readout chain
follows Xue et al., IEEE JSSC 2020.

Importing this package registers :class:`IsubIadc1t1rCimMacro` under its
config key.
"""

from neurox.works.macro.cim.isub_iadc_1t1r import macro

__all__ = ["macro"]
