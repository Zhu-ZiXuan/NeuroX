# Current DAC family

## Shared conventions

A current DAC converts an unsigned integer code $c\in[0,c_{\max}]$ to a single-ended output current $I_{\mathrm{out}}$. Output and noise use uA. Each member specifies its transfer and non-idealities.

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $c$ | input code (non-negative integer) | — | `convert` input |
| $c_{\max}$ | maximum valid code (inclusive) | — | `code_max` |
| $I_{\mathrm{out}}$ | output drive current | uA | `convert` output |

## Assumptions, scope & validity

Out-of-range codes are undefined and are not clamped.

TODO (domain author): the validity range of the code-to-current abstraction (settling, finite output impedance, the compliance-voltage headroom the driven node must leave) common to the family.

## References

TODO.
