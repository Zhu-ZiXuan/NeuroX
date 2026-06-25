# DAC Abstract Layer

## Summary / role

Every concrete DAC in the family converts an integer code into an analog drive voltage under one shared contract: a code-to-voltage conversion and a code-to-nominal-voltage lookup table. In the crossbar a DAC produces the word-line drive voltage from the integer activation code; the conversion is the input boundary of the array operating-point solve. This document specifies the family contract; each concrete transfer characteristic is in its own document (see the [DAC family](README.md) index).

## Conversion contract

A DAC maps a non-negative integer code to an analog voltage. The family exposes two surfaces:

- the per-call conversion that produces the drive voltage (with any optional drive noise applied);
- the code-to-signal lookup table - the nominal voltage per code, read directly to obtain an operating-point voltage without invoking the noisy conversion.

The lookup table has one entry per legal code; its length is the code count the DAC supports.

## Governing equations

The conversion is a table lookup perturbed by optional drive noise,

$$V_{\mathrm{out}} = L[\,\mathrm{code}\,] + n,$$

where $L$ is the code-to-signal LUT and $n$ the optional per-call drive-noise sample ($n = 0$ with the policy off). The nominal LUT read returns $L[\mathrm{code}]$ exactly. An out-of-range input code is clamped (saturated) to the representable code range $[0,\ \text{code count} - 1]$ before the lookup, consistent with the ADC clamp and conventional DAC saturation.

## Numerical method

N/A - the conversion is a single table lookup per call; no iteration.

## Noise & non-idealities

The abstract layer fixes no noise source; each concrete DAC declares its own. The DAC's per-element switching energy is captured by a per-op energy parameter on the concrete config and is set to zero whenever the same energy is accounted at another stage, to avoid double-counting.

## Parameters

The abstract layer carries no physical parameter beyond the static-PPA fields (area, leakage) inherited by every member. Per-impl parameters are tabulated in the concrete documents. Provenance terms are defined in [parameter_provenance](../../parameter_provenance.md).

## Assumptions, scope & validity

Stated assumption: the conversion is a pure code-indexed lookup; the DAC introduces no code-dependent nonlinearity beyond what the LUT entries encode.

TODO (domain author): the validity range of the lookup abstraction (settling, finite output impedance under load) common to the family.

## Validation

TODO - link validation evidence once written.

## References

TODO.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $L$ | code-to-signal lookup table | V | `code_to_signal` |
| $V_{\mathrm{out}}$ | output drive voltage | V | `convert` output |
| $n$ | per-call drive-noise sample | V | sampled in `convert` |

---

- **Internals**: [dac base internals](../../../internals/analog/dac/base.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `DACConfig` (see `api`)
- **Decisions**: N/A — no ADR governs this module.
