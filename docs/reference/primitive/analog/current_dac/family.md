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

Settling, output impedance, and compliance limits depend on the concrete converter.
