# Current reference

## Physical model

A multi-output current reference holds one strictly increasing row of nominal taps per operating mode. Each fabricated instance carries its own initial-accuracy spread. The bias-generation topology is not modelled: its power is represented as a static, always-on draw independent of tap values. Two departures from nominal are modelled: per-instance initial accuracy fixed at fabrication and per-read thermal/flicker noise. Both are relative to the nominal tap.

## Governing equations

The $k$-th sourced tap of mode row $m$ is the nominal value scaled by the two relative departures,

$$I_{\mathrm{ref},m,k} = I_{\mathrm{ref},m,k}^{\mathrm{nom}} \, (1 + \delta_{m,k})(1 + \eta_{m,k}),$$

where $\delta_{m,k} \sim \mathcal{N}(0, \sigma_{\mathrm{tol}}^2)$ is the per-instance initial-accuracy spread, fixed once at fabrication, and $\eta_{m,k} \sim \mathcal{N}(0, \sigma_{\mathrm{noise}}^2)$ is the per-read noise, resampled every read. Both $\delta_{m,k}$ and $\eta_{m,k}$ are zero-mean, so the mean tap is the nominal $I_{\mathrm{ref},m,k}^{\mathrm{nom}}$.

## Numerical method

N/A — each tap is a closed-form per-call sample; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| initial accuracy | per-die process variation + trim residual of the reference | relative (multiplicative) zero-mean Gaussian on each tap, fixed once per instance at fabrication | tolerance sigma |
| reference noise | thermal / flicker fluctuation on the reference | relative (multiplicative) zero-mean Gaussian on each tap, resampled per read | noise sigma |

TODO (domain author): give each sigma's physical derivation and citation, and confirm whether a temperature-drift term (correlated, not per-read i.i.d.) should be modelled separately.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `i_refs__uA` ($I_{\mathrm{ref},m,k}^{\mathrm{nom}}$) | nominal reference-current tap rows, `[mode][tap]` (one or more equal-length rows) | uA | $\geq 0$, strictly increasing per row | Design |
| `tolerance_sigma_relative` ($\sigma_{\mathrm{tol}}$) | relative initial-accuracy sigma, fixed at fabrication | — | $\geq 0$ | Measured |
| `noise_sigma_relative` ($\sigma_{\mathrm{noise}}$) | relative per-read noise sigma | — | $\geq 0$ | Measured |
| leakage / area | static PPA / spec fields (leakage carries all static power, incl. the always-on bias network that generates the currents) | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $I_{\mathrm{ref},m,k}^{\mathrm{nom}}$ | nominal reference-current tap rows, `[mode][tap]` | uA | `i_refs__uA` |
| $I_{\mathrm{ref},m,k}$ | actual sourced taps (post tolerance + noise), full per-instance bank `[*inst, mode, tap]` | uA | `CurrentReferenceSnap.i_refs__uA` |
| $\sigma_{\mathrm{tol}}$ | relative initial-accuracy sigma | — | `tolerance_sigma_relative` |
| $\sigma_{\mathrm{noise}}$ | relative per-read noise sigma | — | `noise_sigma_relative` |

## Assumptions, scope & validity

Stated assumption: the bias power that generates the reference currents is a constant static draw, not derived from the tap values; the initial accuracy is a per-die constant and the reference noise is per-read i.i.d., both relative to the nominal tap. Each tap is non-negative; a `0` uA tap is valid and denotes a ground/rail reference (relative noise scales it to exactly `0`, so it stays stable and correct). Mode selection is quasi-static: switching the active tap row is far slower than a conversion, so no per-conversion switching energy is modelled. The model is valid where the bias power is well-approximated as constant over the operating range.

TODO (domain author): whether slow temperature drift within an inference must be modelled as a correlated term rather than per-read noise, and whether a tap-current-dependent bias-power term is needed for designs where the sourced currents dominate the static draw.

## Validation

TODO — link validation evidence once written.

## References

TODO: cite the reference-accuracy and drift models.

---

- **Internals**: [current_reference internals](../../../internals/primitive/analog/current_reference.md)
- **Validation**: TODO — validation evidence not yet written
- **Configuration**: `CurrentReferenceConfig`, `CurrentReferencePolicy` (see `api`)
