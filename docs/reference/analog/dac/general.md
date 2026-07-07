# General DAC

## Summary / role

`GeneralDAC` is the LUT DAC: a code-to-voltage lookup table plus additive Gaussian drive-thermal noise. It is a member of the [DAC family](family.md) and honours the family contract in [base](family.md).

## Physical model

The DAC is modelled as an ideal code-indexed voltage source: an integer code produces the tabulated voltage $L[\mathrm{code}]$, and the single modelled non-ideality is additive Gaussian drive-thermal noise on the output, sampled per conversion.

## Governing equations

$$V_{\mathrm{out}} = L[\,\mathrm{code}\,] + n,\qquad n \sim \mathcal{N}(0,\ \sigma_{\mathrm{drive}}^2),$$

with $L$ the `code_to_signal` LUT and $n$ the per-conversion drive-thermal sample. The nominal LUT read returns $L[\mathrm{code}]$ exactly.

## Numerical method

N/A - a single table lookup per call.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| drive thermal | thermal noise on the drive output | additive zero-mean Gaussian, standard deviation $\sigma_{\mathrm{drive}}$ | `drive_thermal__V` |

The drive-thermal source is dynamic (per conversion); `GeneralDAC` carries no static mismatch.

TODO (domain author): physical derivation and citation for the drive-thermal sigma.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `code_to_signal` | LUT entry per integer code | V | Design |
| `drive_thermal__V` | additive drive-thermal noise standard deviation | V | Measured |
| `energy_per_op__fJ` | per-conversion dynamic energy | fJ | Design |
| leakage / area / latency | static PPA / spec fields | uW, um^2, ns | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $L$ | code-to-signal LUT | V | `code_to_signal` |
| $V_{\mathrm{out}}$ | output drive voltage | V | `convert` output |
| $n$ | drive-thermal noise sample | V | sampled in `convert` |
| $\sigma_{\mathrm{drive}}$ | drive-thermal noise standard deviation | V | `drive_thermal__V` |

## Assumptions, scope & validity

Stated assumption: the output is a pure LUT lookup plus additive Gaussian noise; no code-dependent nonlinearity beyond the LUT entries.

TODO (domain author): the validity range of the ideal-LUT abstraction (settling, output impedance under load).

## Validation

TODO - link validation evidence once written.

## References

TODO.

---

- **Internals**: [general internals](../../../internals/analog/dac/general.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `GeneralDACConfig`, `GeneralDACPolicy` (see `api`)
