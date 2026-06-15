# Analog Mux

## Summary / role

The `AnalogMux` is a leaf analog transport block in the readout chain: it moves a grouped differential signal from one stage to the next, applying a scalar transport gain and optionally injecting transport noise. It only transports and tallies the transport energy. It is a single concrete block a consuming circuit composes directly, not a polymorphic family.

## Physical model

Physically, the mux transports the differential signal from one stage to the next through a switched path. A real switch path has an insertion gain or loss, so the transported pair is scaled relative to its input rather than passed through unchanged.

This block represents that transport with a single scalar gain $g$ (`mux_gain`) applied identically to both legs. The single-scalar-gain model is a first-order approximation: it ignores the frequency response of the switch path, its nonlinearity, and any inter-leg mismatch in insertion gain. The inter-leg mismatch is instead captured separately by the differential-mode noise term (`mux_noise_dm`). Under this approximation the only further departures from ideal are two optional additive noise terms, one acting on the common mode and one on the differential mode of the transported pair.

TODO (domain author): confirm the scalar-gain model.

## Governing equations

For a differential input pair each leg is scaled by the transport gain $g$ and perturbed by the two optional noise terms,

$$V_{\mathrm{out}}^{\pm} = g\,V_{\mathrm{in}}^{\pm} + n_{\mathrm{cm}} \pm n_{\mathrm{dm}},$$

where $g$ is the transport gain (`mux_gain`), $n_{\mathrm{cm}}$ is the common-mode noise sample (shared by both legs) and $n_{\mathrm{dm}}$ the differential-mode noise sample (added in full to the positive leg and subtracted in full from the negative leg); each noise term is zero when its policy switch is off.

TODO (domain author): confirm the exact decomposition (whether DM noise enters as written or directly on the differential value) against the transport kernel, and give each term's distribution.

## Numerical method

N/A - the transport is a closed-form per-call map; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter | Policy switch |
|---|---|---|---|---|
| common-mode noise | transport-path common-mode disturbance | additive zero-mean Gaussian on both legs | CM noise sigma | `mux_noise_cm` |
| differential-mode noise | transport-path differential disturbance | additive zero-mean Gaussian, added to the positive leg and subtracted from the negative leg | DM noise sigma | `mux_noise_dm` |

Both noise sources are dynamic (sampled per transport); the mux carries no static mismatch.

TODO (domain author): physical origin and citation for each noise sigma, and any temperature scaling.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `mux_gain` ($g$) | scalar transport gain applied to both legs | — | Design |
| CM noise sigma | common-mode transport-noise sigma | V | Measured |
| DM noise sigma | differential-mode transport-noise sigma | V | Measured |
| transport energy | per-transport dynamic energy | fJ | Design |
| leakage / area / latency | static PPA / spec fields | uW, um^2, ns | Design |

Provenance terms are defined in [parameter_provenance](../parameter_provenance.md).

## Assumptions, scope & validity

Stated assumption: the mux is a scalar-gain differential pass element that scales each leg by $g$; all further departures are the two additive noise terms above.

First-order transport-gain approximation: the single scalar $g$ applied identically to both legs is a modelling approximation. It ignores the frequency response of the switch path, its nonlinearity, and inter-leg mismatch in insertion gain; the inter-leg mismatch is captured separately by the differential-mode noise term (`mux_noise_dm`). The approximation is valid where the transported signal sits within the switch path's linear, settled band.

TODO (domain author): neglected charge-injection / settling / channel-resistance effects and the validity range of the ideal-pass abstraction.

## Validation

TODO - link validation evidence once written.

## References

TODO.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $g$ | scalar transport gain | — | `mux_gain` |
| $V_{\mathrm{in}}^{\pm}, V_{\mathrm{out}}^{\pm}$ | input / output differential legs | V | transport input / output |
| $n_{\mathrm{cm}}$ | common-mode noise sample | V | sampled in transport |
| $n_{\mathrm{dm}}$ | differential-mode noise sample | V | sampled in transport |

---

- **Internals**: [analog_mux internals](../../internals/analog/analog_mux.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `AnalogMuxConfig` (see `api`)
- **Decisions**: N/A — no ADR governs this module.
