# General voltage DAC

## Physical model

An ideal code-indexed voltage source follows the [voltage DAC family's](family.md) unsigned code domain. Its transfer is a lookup table plus additive Gaussian drive-thermal noise sampled per conversion.

## Governing equations

$$V_{\mathrm{out}} = L[\,\mathrm{code}\,] + n,\qquad n \sim \mathcal{N}(0,\ \sigma_{\mathrm{drive}}^2),$$

Here $L$ is the nominal code-to-voltage LUT and $n$ the drive-thermal sample.

One conversion to code $c$ draws dynamic energy $E[c]$ from a table parallel to $L$. A zero entry denotes a free drive event.

## Noise & non-idealities

| [Source](../../../../conventions/module_parameter.md) | Physical origin | Statistical model |
| --- | --- | --- |
| drive thermal | thermal noise on the drive output | additive zero-mean Gaussian, standard deviation $\sigma_{\mathrm{drive}}$ |

The noise spread is a supplied characterization parameter, constant across codes.

## Parameters

| Parameter | Meaning | Unit | Constraint | [Source](../../../../conventions/module_parameter.md) |
| --- | --- | --- | --- | --- |
| `code_to_signal` ($L$) | LUT entry per integer code | V | — | Design |
| `drive_thermal__V` ($\sigma_{\mathrm{drive}}$) | additive drive-thermal noise standard deviation | V | $\geq 0$ | Measured |
| `code_to_per_op_energy__fJ` ($E$) | per-code conversion energy | fJ | same length as $L$; finite entries $\geq 0$ | Design |
| leakage / area | static PPA / spec fields | uW, um^2 | $\geq 0$ | Design |

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

The LUT and noise parameters require characterization over the intended load and operating range; no measured envelope is specified here.
