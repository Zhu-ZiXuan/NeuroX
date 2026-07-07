# DAC family

## Summary / role

Every concrete DAC in the family converts a non-negative integer code into an analog drive voltage under one shared contract: a per-call code-to-voltage conversion over the member's valid code range. This document specifies the family contract; each concrete transfer characteristic and noise model is in its own document.

## Conversion contract

A DAC maps a non-negative integer code to the analog drive voltage $V_{\mathrm{out}}$. The family fixes the valid code range: valid codes run from 0 to `code_max` (inclusive), the maximum valid code each member reports, and the conversion is defined only on codes in that range. The realized output is the member's nominal code-to-voltage transfer modified by its own declared non-idealities; the family fixes neither the transfer form nor a noise model.

## Governing equations

N/A at the family level - the family fixes no transfer form. Each concrete member specifies its own code-to-voltage transfer and noise.

## Numerical method

N/A at the family level - the conversion carries no iteration at the shared layer; each concrete member specifies its own method.

## Noise & non-idealities

The abstract layer fixes no noise source; each concrete DAC declares its own.

## Parameters

The abstract layer carries no physical parameter beyond the static-PPA fields (area, leakage) inherited by every member. Per-impl parameters are tabulated in the concrete documents. Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Assumptions, scope & validity

Stated assumption: the input code lies in the valid range 0 to `code_max`; the family does not itself saturate an out-of-range code.

TODO (domain author): the validity range of the code-to-voltage abstraction (settling, finite output impedance under load) common to the family.

## Validation

TODO - link validation evidence once written.

## References

TODO.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{out}}$ | output drive voltage | V | `convert` output |

---

- **Internals**: [dac base internals](../../../internals/analog/dac/base.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `DACConfig` (see `api`)
