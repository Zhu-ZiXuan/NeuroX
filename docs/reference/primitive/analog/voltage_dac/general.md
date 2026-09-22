# General voltage DAC

## Physical model

An ideal code-indexed voltage source follows the [voltage DAC family's](family.md) unsigned code domain. Its transfer is a lookup table plus additive Gaussian drive-thermal noise sampled per conversion.

## Governing equations

$$V_{\mathrm{out}} = L[\,\mathrm{code}\,] + n,\qquad n \sim \mathcal{N}(0,\ \sigma_{\mathrm{drive}}^2),$$

Here $L$ is the nominal code-to-voltage LUT and $n$ the drive-thermal sample.

One conversion to code $c$ draws dynamic energy $E[c]$ from a table parallel to $L$. A zero entry denotes a free drive event.

## Numerical method

N/A — direct table lookup, no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model |
| --- | --- | --- |
| drive thermal | thermal noise on the drive output | additive zero-mean Gaussian, standard deviation $\sigma_{\mathrm{drive}}$ |

TODO (domain author): physical derivation and citation for the drive-thermal sigma.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
| --- | --- | --- | --- | --- |
| `code_to_signal` ($L$) | LUT entry per integer code | V | — | Design |
| `drive_thermal__V` ($\sigma_{\mathrm{drive}}$) | additive drive-thermal noise standard deviation | V | $\geq 0$ | Measured |
| `code_to_per_op_energy__fJ` ($E$) | per-code conversion energy | fJ | same length as $L$; finite entries $\geq 0$ | Design |
| leakage / area | static PPA / spec fields | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $L$ | code-to-signal LUT | V | `code_to_signal` |
| $E$ | per-code per-op energy LUT | fJ | `code_to_per_op_energy__fJ` |
| $V_{\mathrm{out}}$ | output drive voltage | V | `convert` output |
| $n$ | drive-thermal noise sample | V | sampled in `convert` |
| $\sigma_{\mathrm{drive}}$ | drive-thermal noise standard deviation | V | `drive_thermal__V` |

## Assumptions, scope & validity

Every instance shares the exact nominal LUT, with no static mismatch or nonlinearity beyond its entries.

TODO (domain author): the validity range of the ideal-LUT abstraction (settling, output impedance under load).

## Validation

TODO - link validation evidence once written.

## References

TODO.
