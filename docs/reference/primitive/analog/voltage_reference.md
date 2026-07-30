# Voltage reference

## Physical model

An ideal multi-output voltage source holding one set of fixed nominal voltage taps per operating mode, the modes being equal in length and non-negative. Ordering within a mode is not a property of the source: an ascending decision ladder and an unordered set of bias taps are equally valid modes, so the meaning of a mode is fixed by the circuit the taps drive, not by the reference. A single-mode single-tap bank — one clamp voltage — is the degenerate case of the same model. The taps sit on ideal high-impedance nodes that draw no signal current, so there is no data-dependent dissipation — the entire hardware cost is static: the always-on bias power and the silicon area. The bias topology that sets the taps is not modelled. Two departures from the nominal taps are modelled: a static per-instance initial-accuracy spread (process variation plus trim residual) fixed at fabrication, and a per-read noise on the reference node (thermal / flicker). Both are relative to the nominal tap so a single sigma applies uniformly across taps of differing magnitude.

## Governing equations

The $k$-th sourced tap of mode $m$ is the nominal value scaled by the two relative departures,

$$V_{\mathrm{ref},m,k} = V_{\mathrm{ref},m,k}^{\mathrm{nom}} \, (1 + \delta_{m,k})(1 + \eta_{m,k}),$$

where $\delta_{m,k} \sim \mathcal{N}(0, \sigma_{\mathrm{tol}}^2)$ is the per-instance initial-accuracy spread, fixed once at fabrication, and $\eta_{m,k} \sim \mathcal{N}(0, \sigma_{\mathrm{noise}}^2)$ is the per-read noise, resampled every read. $\delta_{m,k}$ and $\eta_{m,k}$ are zero-mean, so the mean tap is the nominal $V_{\mathrm{ref},m,k}^{\mathrm{nom}}$.

## Numerical method

N/A — each tap is a closed-form per-call sample; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| initial accuracy | per-die process variation + trim residual of the reference | relative (multiplicative) zero-mean Gaussian on each tap, fixed once per instance at fabrication | tolerance sigma |
| reference noise | thermal / flicker fluctuation on the reference node | relative (multiplicative) zero-mean Gaussian on each tap, resampled per read | noise sigma |

The two sources are drawn over different domains, and both statements hold at once because they describe different physical quantities. The initial-accuracy spread is indexed by fabricated instance: one reference generator per instance, one draw held for its lifetime. The reference noise is indexed by every tap value a read produces — a reference consumed at $N$ distinct instants, or at $N$ distinct points of the distribution net, carries $N$ independent draws, because thermal and flicker fluctuation is neither reused across instants nor shared between nodes.

TODO (domain author): give each sigma's physical derivation and citation, and confirm whether a temperature-drift term (correlated, not per-read i.i.d.) should be modelled separately.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `v_refs__V` ($V_{\mathrm{ref},m,k}^{\mathrm{nom}}$) | nominal reference-voltage taps, `[mode][tap]` (one or more equal-length modes) | V | $\geq 0$; equal tap lengths; no ordering within a mode | Design |
| `tolerance_sigma_relative` ($\sigma_{\mathrm{tol}}$) | relative initial-accuracy sigma, fixed at fabrication | — | $\geq 0$ | Measured |
| `noise_sigma_relative` ($\sigma_{\mathrm{noise}}$) | relative per-read noise sigma | — | $\geq 0$ | Measured |
| leakage / area | static PPA / spec fields (leakage carries all static power, incl. the always-on bias network) | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{ref},m,k}^{\mathrm{nom}}$ | nominal reference-voltage taps, `[mode][tap]` | V | `v_refs__V` |
| $V_{\mathrm{ref},m,k}$ | actual sourced taps of the selected mode (post tolerance + noise), one value per read position | V | `VrefSnap.v_refs__V` |
| $\sigma_{\mathrm{tol}}$ | relative initial-accuracy sigma | — | `tolerance_sigma_relative` |
| $\sigma_{\mathrm{noise}}$ | relative per-read noise sigma | — | `noise_sigma_relative` |

## Assumptions, scope & validity

Stated assumption: the taps are ideal high-impedance nodes (no load-dependent droop, no signal current), so the block's cost is purely static; the initial accuracy is a per-die constant and the reference noise is i.i.d. across reads and across the distinct instants and nodes within one read, both relative to the nominal tap. Each tap is non-negative; a `0` V tap is valid and denotes a ground/rail reference (relative noise scales it to exactly `0`, so it stays stable and correct). Mode selection is quasi-static: switching the active mode is far slower than a read, so no per-read switching energy is modelled. A static departure belonging to the circuit that reads the tap — a comparator offset, a driver offset — is a property of that circuit and is modelled there, so the reference carries no second static spread per reading site; counting it in both places would double the same physical deviation. The model is valid where the reference distribution load is negligible and the bias power is well-approximated as a constant.

TODO (domain author): the validity boundary of the load-free idealisation, and whether slow temperature drift within an inference must be modelled as a correlated term rather than per-read noise.

## Validation

TODO — link validation evidence once written.

## References

TODO: cite the reference-accuracy and drift models.

---

- **Internals**: [voltage_reference internals](../../../internals/primitive/analog/voltage_reference.md)
- **Validation**: TODO — validation evidence not yet written
- **Configuration**: `VrefConfig`, `VrefPolicy` (see `api`)
