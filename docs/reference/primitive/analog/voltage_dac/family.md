# Voltage DAC family

## Shared conventions

A voltage DAC converts a non-negative integer code $c$ to an analog drive voltage $V_{\mathrm{out}}$. The code domain is unsigned: valid codes run from $0$ to $c_{\max}$ inclusive — the maximum valid code each member reports — and the conversion is defined only on that range. The converter commits to the voltage domain: the output and every noise term are expressed in volts. The realized output is the member's nominal code-to-voltage transfer perturbed by its own non-idealities; the family fixes the unsigned code domain and the code-to-voltage direction, leaving the transfer shape and any noise to each member.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $c$ | input code (non-negative integer) | — | `convert` input |
| $c_{\max}$ | maximum valid code (inclusive) | — | `code_max` |
| $V_{\mathrm{out}}$ | output drive voltage | V | `convert` output |

## Assumptions, scope & validity

Stated assumption: the input code lies in the valid range $[0,\ c_{\max}]$. The conversion is defined only on that range and does not clamp an out-of-range code.

TODO (domain author): the validity range of the code-to-voltage abstraction (settling, finite output impedance under load) common to the family.

## References

TODO.

---

- **Internals**: [voltage DAC base internals](../../../../internals/primitive/analog/voltage_dac/base.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `VoltageDacConfig` (see `api`)
