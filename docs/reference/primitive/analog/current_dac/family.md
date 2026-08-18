# Current DAC family

## Shared conventions

A current DAC converts a non-negative integer code $c$ to a single-ended analog output current $I_{\mathrm{out}}$. The code domain is unsigned: valid codes run from $0$ to $c_{\max}$ inclusive — the maximum valid code each member reports — and the conversion is defined only on that range. The converter commits to the current domain: the output and every noise term are expressed in microamperes. The realized output is the member's nominal code-to-current transfer perturbed by its own non-idealities; the family fixes the unsigned code domain and the code-to-current direction, leaving the transfer shape and any noise to each member.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $c$ | input code (non-negative integer) | — | `convert` input |
| $c_{\max}$ | maximum valid code (inclusive) | — | `code_max` |
| $I_{\mathrm{out}}$ | output drive current | uA | `convert` output |

## Assumptions, scope & validity

Stated assumption: the input code lies in the valid range $[0,\ c_{\max}]$. The conversion is defined only on that range and does not clamp an out-of-range code.

TODO (domain author): the validity range of the code-to-current abstraction (settling, finite output impedance, the compliance-voltage headroom the driven node must leave) common to the family.

## References

TODO.
