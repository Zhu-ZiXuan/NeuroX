# General DAC

## Summary / role

`GeneralDAC` is the simplest concrete DAC: a code-to-voltage lookup table plus optional additive Gaussian drive-thermal noise. It is the only current member of the [DAC family](README.md) and honours the family contract in [base](base.md).

## Physical model

The DAC is modelled as an ideal code-indexed voltage source: code $c$ produces the tabulated voltage $L[c]$, and the single optional non-ideality is additive Gaussian drive-thermal noise on the output, sampled per conversion.

## Governing equations

$$V_{\mathrm{out}} = L[\,\mathrm{code}\,] + n,\qquad n \sim \mathcal{N}(0,\ \sigma_{\mathrm{drive}}^2),$$

with $L$ the `code_to_signal` LUT and $n$ the drive-thermal sample (zero with the policy off). The nominal LUT read returns $L[\mathrm{code}]$ exactly.

## Numerical method

N/A - a single table lookup per call.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter | Policy switch |
|---|---|---|---|---|
| drive thermal | thermal noise on the drive output | additive zero-mean Gaussian, sigma $\sigma_{\mathrm{drive}}$ | `drive_thermal__V` | `drive_thermal` |

The drive-thermal source is dynamic (per convert); `GeneralDAC` carries no static mismatch.

TODO (domain author): physical derivation and citation for the drive-thermal sigma.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `code_to_signal` | LUT entry per integer code | V | Design |
| `drive_thermal__V` | additive drive-thermal noise sigma | V | Measured |
| `energy_per_op__fJ` | per-conversion dynamic energy | fJ | Design |
| leakage / area / latency | static PPA / spec fields | uW, um^2, ns | Design |

Set `energy_per_op__fJ` to zero whenever the same switching energy is accounted at another stage, to avoid double-counting. Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Assumptions, scope & validity

Stated assumption: the output is a pure LUT lookup plus additive Gaussian noise; no code-dependent nonlinearity beyond the LUT entries.

TODO (domain author): the validity range of the ideal-LUT abstraction (settling, output impedance under load).

## Validation

TODO - link validation evidence once written.

## References

TODO.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $L$ | code-to-signal LUT | V | `code_to_signal` |
| $V_{\mathrm{out}}$ | output drive voltage | V | `convert` output |
| $n$ | drive-thermal noise sample | V | sampled in `convert` |
| $\sigma_{\mathrm{drive}}$ | drive-thermal noise sigma | V | `drive_thermal__V` |

---

- **Internals**: [general internals](../../../internals/analog/dac/general.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `GeneralDACConfig`, `GeneralDACPolicy` (see `api`)
- **Decisions**: N/A — no ADR governs this module.
