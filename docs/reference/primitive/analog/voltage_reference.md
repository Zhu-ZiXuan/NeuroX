# Voltage reference

## Physical model

An ideal multi-output voltage source holds one set of fixed nominal taps per operating mode; modes have equal positive lengths, taps are non-negative, and their ordering is unconstrained. A single-mode single-tap bank is the degenerate case. The taps sit on ideal high-impedance nodes that draw no signal current, so the entire hardware cost is static: the always-on bias power and silicon area. The bias topology that sets the taps is not modeled. The modeled departure from nominal is a static per-instance initial-accuracy spread fixed at fabrication; access-varying fluctuations are outside the model.

## Governing equations

The $k$-th sourced tap of mode $m$ is the nominal value scaled by the relative departure,

$$V_{\mathrm{ref},m,k} = V_{\mathrm{ref},m,k}^{\mathrm{nom}} \, (1 + \delta_{m,k}),$$

where $\delta_{m,k} \sim \mathcal{N}(0, \sigma_{\mathrm{tol}}^2)$ is the per-instance initial-accuracy spread, fixed once at fabrication. $\delta_{m,k}$ is zero-mean, so the mean tap is the nominal $V_{\mathrm{ref},m,k}^{\mathrm{nom}}$.

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
| `v_refs__V` ($V_{\mathrm{ref},m,k}^{\mathrm{nom}}$) | nominal reference-voltage taps, `[mode][tap]` (one or more equal-length modes) | V | $\geq 0$; equal tap lengths; no ordering within a mode | Design |
| `tolerance_sigma_relative` ($\sigma_{\mathrm{tol}}$) | relative initial-accuracy sigma, fixed at fabrication | — | $\geq 0$ | Measured |
| leakage / area | static PPA / spec fields (leakage carries all static power, incl. the always-on bias network) | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{ref},m,k}^{\mathrm{nom}}$ | nominal reference-voltage taps, `[mode][tap]` | V | `v_refs__V` |
| $V_{\mathrm{ref},m,k}$ | actual sourced taps (post tolerance), one bank per fabricated instance | V | `v_out__V` |
| $\sigma_{\mathrm{tol}}$ | relative initial-accuracy sigma | — | `tolerance_sigma_relative` |

## Assumptions, scope & validity

Stated assumption: the taps are ideal high-impedance nodes with no load-dependent droop or signal current, so the block's cost is purely static; the initial accuracy is a per-die constant relative to the nominal tap, and the sourced value is time-invariant between fabrications. Each tap is non-negative; a `0` V tap is valid and remains exactly zero under relative tolerance. Mode selection is quasi-static, so no switching energy is modeled. The model is valid where the reference distribution load is negligible and the bias power is well-approximated as constant.

TODO (domain author): the validity boundary of the load-free idealisation, and whether slow temperature drift within an inference must be modelled as a correlated term on the source.

## Validation

TODO — link validation evidence once written.

## References

TODO: cite the reference-accuracy and drift models.
