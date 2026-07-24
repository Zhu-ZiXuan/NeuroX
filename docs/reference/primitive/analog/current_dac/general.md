# General current DAC

The simplest member of the [current DAC family](family.md): a code-steered current source whose transfer is a tabulated lookup table, obeying the family unsigned-code domain.

## Physical model

The DAC is modelled as an ideal code-steered current source: an integer code selects the tabulated output current $L[\mathrm{code}]$, and the single modelled non-ideality is additive Gaussian noise on the steered output, sampled per conversion.

## Governing equations

$$I_{\mathrm{out}} = L[\,\mathrm{code}\,] + n,\qquad n \sim \mathcal{N}(0,\ \sigma_{\mathrm{drive}}^2),$$

with $L$ the code-to-current LUT and $n$ the per-conversion output-noise sample. The nominal LUT read returns $L[\mathrm{code}]$ exactly.

## Numerical method

N/A — direct table lookup, no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model |
|---|---|---|
| drive thermal | thermal noise on the steered output current | additive zero-mean Gaussian, standard deviation $\sigma_{\mathrm{drive}}$, constant across codes |

The drive-thermal source is dynamic, resampled every conversion; the model carries no static mismatch.

TODO (domain author): physical derivation and citation for the drive-thermal sigma, and the code-dependent output-noise law of the steered current sources.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `code_to_signal` ($L$) | LUT entry per integer code | uA | — | Design |
| `drive_thermal__uA` ($\sigma_{\mathrm{drive}}$) | additive output-noise standard deviation | uA | $\geq 0$ | Measured |
| `energy_per_op__fJ` | per-conversion dynamic energy | fJ | $\geq 0$ | Design |
| `latency_per_op__ns` | per-conversion latency | ns | $\geq 0$ | Design |
| leakage / area | static PPA / spec fields | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $L$ | code-to-signal LUT | uA | `code_to_signal` |
| $I_{\mathrm{out}}$ | output drive current | uA | `convert` output |
| $n$ | output-noise sample | uA | sampled in `convert` |
| $\sigma_{\mathrm{drive}}$ | output-noise standard deviation | uA | `drive_thermal__uA` |

## Assumptions, scope & validity

Stated assumptions:

- The output is a pure LUT lookup plus additive Gaussian noise; no code-dependent nonlinearity beyond the LUT entries.
- The output noise is signal-independent: one constant sigma applies at every code. The dominant output noise of a real steered current source instead grows with the steered current — $\sigma_{\mathrm{drive}} \propto \sqrt{I_{\mathrm{out}}}$ in the shot-noise limit — so a single fitted sigma over-states the noise near zero code and under-states it at full scale.
- The steering is ideal: the tabulated current is delivered in full whatever voltage the driven node settles to, so neither finite output impedance nor the compliance-voltage headroom real steering needs bounds the transfer.

TODO (domain author): the validity range of the ideal-LUT abstraction (settling, output impedance under load).

## Validation

TODO - link validation evidence once written.

## References

TODO.

---

- **Internals**: [general internals](../../../../internals/primitive/analog/current_dac/general.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `GeneralIdacConfig`, `GeneralIdacPolicy` (see `api`)
