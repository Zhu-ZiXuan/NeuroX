# Current reference

## Physical model

A multi-output current reference holds one set of nominal taps per operating mode; modes have equal positive lengths, taps are non-negative, and their ordering is unconstrained. A single-mode single-tap bank is the degenerate case. Each fabricated instance carries its own initial-accuracy spread. The bias-generation topology is not modelled: its power is represented as a static, always-on draw independent of tap values. The modeled departure from nominal is the per-instance initial accuracy fixed at fabrication; access-varying fluctuations are outside the model.

## Governing equations

The $k$-th sourced tap of mode $m$ is the nominal value scaled by the relative departure,

$$I_{\mathrm{ref},m,k} = I_{\mathrm{ref},m,k}^{\mathrm{nom}} \, (1 + \delta_{m,k}),$$

where $\delta_{m,k} \sim \mathcal{N}(0, \sigma_{\mathrm{tol}}^2)$ is the per-instance initial-accuracy spread, fixed once at fabrication. $\delta_{m,k}$ is zero-mean, so the mean tap is the nominal $I_{\mathrm{ref},m,k}^{\mathrm{nom}}$.

## Numerical method

N/A — each tap is a closed-form sample drawn once at fabrication; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| initial accuracy | per-die process variation + trim residual of the reference | relative (multiplicative) zero-mean Gaussian on each tap, fixed once per instance at fabrication | tolerance sigma |

The spread is indexed by fabricated instance: one draw is held unchanged for the instance lifetime. Thermal and flicker fluctuations that vary by access are not modeled.

TODO (domain author): give the sigma's physical derivation and citation, and confirm whether a temperature-drift term (correlated across accesses) should be modelled separately.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `i_refs__uA` ($I_{\mathrm{ref},m,k}^{\mathrm{nom}}$) | nominal reference-current taps, `[mode][tap]` (one or more equal-length modes) | uA | $\geq 0$; equal tap lengths; no ordering within a mode | Design |
| `tolerance_sigma_relative` ($\sigma_{\mathrm{tol}}$) | relative initial-accuracy sigma, fixed at fabrication | — | $\geq 0$ | Measured |
| leakage / area | static PPA / spec fields (leakage carries all static power, incl. the always-on bias network that generates the currents) | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $I_{\mathrm{ref},m,k}^{\mathrm{nom}}$ | nominal reference-current taps, `[mode][tap]` | uA | `i_refs__uA` |
| $I_{\mathrm{ref},m,k}$ | actual sourced taps (post tolerance), one bank per fabricated instance | uA | `i_out__uA` |
| $\sigma_{\mathrm{tol}}$ | relative initial-accuracy sigma | — | `tolerance_sigma_relative` |

## Assumptions, scope & validity

Stated assumption: the bias power that generates the reference currents is a constant static draw, not derived from the tap values; the initial accuracy is a per-die constant relative to the nominal tap, and the sourced value is time-invariant between fabrications. Each tap is non-negative; a `0` uA tap is valid and remains exactly zero under relative tolerance. Mode selection is quasi-static, so no switching energy is modeled. The model is valid where the bias power is well-approximated as constant over the operating range.

TODO (domain author): whether slow temperature drift within an inference must be modelled as a correlated term on the source, and whether a tap-current-dependent bias-power term is needed for designs where the sourced currents dominate the static draw.

## Validation

TODO — link validation evidence once written.

## References

TODO: cite the reference-accuracy and drift models.
