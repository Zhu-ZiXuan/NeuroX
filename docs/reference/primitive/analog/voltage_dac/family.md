# Voltage DAC family

## Shared conventions

A voltage DAC converts an unsigned integer code $c\in[0,c_{\max}]$ to an output voltage $V_{\mathrm{out}}$. Output and noise use V. Each member specifies its transfer and non-idealities.

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $c$ | input code (non-negative integer) | — | `convert` input |
| $c_{\max}$ | maximum valid code (inclusive) | — | `code_max` |
| $V_{\mathrm{out}}$ | output drive voltage | V | `convert` output |

## Assumptions, scope & validity

Out-of-range codes are undefined and are not clamped.

TODO (domain author): the validity range of the code-to-voltage abstraction (settling, finite output impedance under load) common to the family.

## References

TODO.
