# General current DAC

## Physical model

An ideal code-steered current source follows the [current DAC family's](family.md) unsigned code domain. Its transfer is a lookup table plus additive Gaussian noise sampled per conversion.

## Governing equations

$$I_{\mathrm{out}} = L[\,\mathrm{code}\,] + n,\qquad n \sim \mathcal{N}(0,\ \sigma_{\mathrm{drive}}^2),$$

Here $L$ is the nominal code-to-current LUT and $n$ the output-noise sample.

## Noise & non-idealities

| [Source](../../../../conventions/module_parameter.md) | Physical origin | Statistical model |
| --- | --- | --- |
| drive thermal | thermal noise on the steered output current | additive zero-mean Gaussian, standard deviation $\sigma_{\mathrm{drive}}$, constant across codes |

The noise spread is a supplied characterization parameter, constant across codes.

## Parameters

| Parameter | Meaning | Unit | Constraint | [Source](../../../../conventions/module_parameter.md) |
| --- | --- | --- | --- | --- |
| `code_to_signal` ($L$) | LUT entry per integer code | uA | — | Design |
| `drive_thermal__uA` ($\sigma_{\mathrm{drive}}$) | additive output-noise standard deviation | uA | $\geq 0$ | Measured |
| `energy_per_op__fJ` | per-conversion dynamic energy | fJ | $\geq 0$ | Design |
| leakage / area | static PPA / spec fields | uW, um^2 | $\geq 0$ | Design |

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $L$ | code-to-signal LUT | uA | `code_to_signal` |
| $I_{\mathrm{out}}$ | output drive current | uA | `convert` output |
| $n$ | output-noise sample | uA | sampled in `convert` |
| $\sigma_{\mathrm{drive}}$ | output-noise standard deviation | uA | `drive_thermal__uA` |

## Assumptions, scope & validity

- Noise uses one sigma across all codes. Where shot noise dominates, $\sigma_{\mathrm{drive}} \propto \sqrt{I_{\mathrm{out}}}$, so a constant fitted sigma overstates noise near zero and understates it at full scale.
- The current is independent of the driven node's voltage; finite output impedance and compliance headroom are excluded.
- Every instance shares the exact nominal LUT, with no static mismatch or nonlinearity beyond its entries.

The LUT and constant-noise approximation require characterization over the intended operating range; no measured envelope is specified here.
