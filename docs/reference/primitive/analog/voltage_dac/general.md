# General voltage DAC

The simplest member of the [voltage DAC family](family.md): a code-indexed voltage source whose transfer is a tabulated lookup table, obeying the family unsigned-code domain.

## Physical model

The DAC is modelled as an ideal code-indexed voltage source: an integer code produces the tabulated voltage $L[\mathrm{code}]$, and the single modelled non-ideality is additive Gaussian drive-thermal noise on the output, sampled per conversion.

## Governing equations

$$V_{\mathrm{out}} = L[\,\mathrm{code}\,] + n,\qquad n \sim \mathcal{N}(0,\ \sigma_{\mathrm{drive}}^2),$$

with $L$ the code-to-voltage LUT and $n$ the per-conversion drive-thermal sample. The nominal LUT read returns $L[\mathrm{code}]$ exactly.

The dynamic energy of one conversion follows the same lookup: converting one element to code $c$ draws $E[c]$, a table parallel to $L$ that states each level's own drive event. A level that costs nothing carries $E[c] = 0$, which is a stated cost and not a missing one.

## Numerical method

N/A — direct table lookup, no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model |
|---|---|---|
| drive thermal | thermal noise on the drive output | additive zero-mean Gaussian, standard deviation $\sigma_{\mathrm{drive}}$ |

The drive-thermal source is dynamic, resampled every conversion; the model carries no static mismatch.

TODO (domain author): physical derivation and citation for the drive-thermal sigma.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `code_to_signal` ($L$) | LUT entry per integer code | V | — | Design |
| `drive_thermal__V` ($\sigma_{\mathrm{drive}}$) | additive drive-thermal noise standard deviation | V | $\geq 0$ | Measured |
| `code_to_per_op_energy__fJ` ($E$) | dynamic energy of converting one element, per integer code — one entry per entry of $L$ | fJ | same length as `code_to_signal`; entries finite, $\geq 0$ (zero is a legitimate cost for a level whose drive event is free) | Design |
| leakage / area | static PPA / spec fields | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $L$ | code-to-signal LUT | V | `code_to_signal` |
| $E$ | per-code per-op energy LUT | fJ | `code_to_per_op_energy__fJ` |
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
