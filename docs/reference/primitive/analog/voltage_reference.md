# Voltage reference

## Physical model

An ideal multi-output voltage source holding one or more fixed nominal voltage taps. The taps sit on ideal high-impedance nodes that draw no signal current, so there is no data-dependent dissipation — the entire hardware cost is static: the always-on bias power and the silicon area. The bias topology that sets the taps is not modelled. Two departures from the nominal taps are modelled: a static per-instance initial-accuracy spread (process variation plus trim residual) fixed at fabrication, and a per-read noise on the reference node (thermal / flicker). Both are relative to the nominal tap so a single sigma applies uniformly across taps of differing magnitude.

## Governing equations

The $k$-th sourced tap is the nominal value scaled by the two relative departures,

$$V_{\mathrm{ref},k} = V_{\mathrm{ref},k}^{\mathrm{nom}} \, (1 + \delta_k)(1 + \eta_k),$$

where $\delta_k \sim \mathcal{N}(0, \sigma_{\mathrm{tol}}^2)$ is the per-instance initial-accuracy spread, fixed once at fabrication, and $\eta_k \sim \mathcal{N}(0, \sigma_{\mathrm{noise}}^2)$ is the per-read noise, resampled every read. $\delta_k$ and $\eta_k$ are zero-mean, so the mean tap is the nominal $V_{\mathrm{ref},k}^{\mathrm{nom}}$.

## Numerical method

N/A — each tap is a closed-form per-call sample; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| initial accuracy | per-die process variation + trim residual of the reference | relative (multiplicative) zero-mean Gaussian on each tap, fixed once per instance at fabrication | tolerance sigma |
| reference noise | thermal / flicker fluctuation on the reference node | relative (multiplicative) zero-mean Gaussian on each tap, resampled per read | noise sigma |

TODO (domain author): give each sigma's physical derivation and citation, and confirm whether a temperature-drift term (correlated, not per-read i.i.d.) should be modelled separately.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `v_refs__V` ($V_{\mathrm{ref},k}^{\mathrm{nom}}$) | nominal reference-voltage taps (one or more) | V | $\geq 0$ | Design |
| `tolerance_sigma_relative` ($\sigma_{\mathrm{tol}}$) | relative initial-accuracy sigma, fixed at fabrication | — | $\geq 0$ | Measured |
| `noise_sigma_relative` ($\sigma_{\mathrm{noise}}$) | relative per-read noise sigma | — | $\geq 0$ | Measured |
| leakage / area | static PPA / spec fields (leakage carries all static power, incl. the always-on bias network) | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{ref},k}^{\mathrm{nom}}$ | nominal reference-voltage taps | V | `v_refs__V` |
| $V_{\mathrm{ref},k}$ | actual sourced taps (post tolerance + noise) | V | `v_ref__V` |
| $\sigma_{\mathrm{tol}}$ | relative initial-accuracy sigma | — | `tolerance_sigma_relative` |
| $\sigma_{\mathrm{noise}}$ | relative per-read noise sigma | — | `noise_sigma_relative` |

## Assumptions, scope & validity

Stated assumption: the taps are ideal high-impedance nodes (no load-dependent droop, no signal current), so the block's cost is purely static; the initial accuracy is a per-die constant and the reference noise is per-read i.i.d., both relative to the nominal tap. Each tap is non-negative; a `0` V tap is valid and denotes a ground/rail reference (relative noise scales it to exactly `0`, so it stays stable and correct). The model is valid where the reference distribution load is negligible and the bias power is well-approximated as a constant.

TODO (domain author): the validity boundary of the load-free idealisation, and whether slow temperature drift within an inference must be modelled as a correlated term rather than per-read noise.

## Validation

TODO — link validation evidence once written.

## References

TODO: cite the reference-accuracy and drift models.

---

- **Internals**: [voltage_reference internals](../../../internals/primitive/analog/voltage_reference.md)
- **Validation**: TODO — validation evidence not yet written
- **Configuration**: `VoltageReferenceConfig`, `VoltageReferencePolicy` (see `api`)
