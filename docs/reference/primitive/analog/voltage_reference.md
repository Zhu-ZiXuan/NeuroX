# Voltage reference

## Physical model

An ideal multi-output voltage source holding one set of fixed nominal voltage taps per operating mode, the modes being equal in length and non-negative. Ordering within a mode is not a property of the source: an ascending decision ladder and an unordered set of bias taps are equally valid modes, so the meaning of a mode is fixed by the circuit the taps drive, not by the reference. A single-mode single-tap bank — one clamp voltage — is the degenerate case of the same model. The taps sit on ideal high-impedance nodes that draw no signal current, so there is no data-dependent dissipation — the entire hardware cost is static: the always-on bias power and the silicon area. The bias topology that sets the taps is not modelled. One departure from the nominal taps is modelled: a static per-instance initial-accuracy spread (process variation plus trim residual) fixed at fabrication, relative to the nominal tap so a single sigma applies uniformly across taps of differing magnitude. The source holds one physical identity, so it is that identity alone; a fluctuation that differs from one access to the next is a property of the circuit reading the tap and is modelled there.

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

The spread is indexed by fabricated instance: one reference generator per instance, one draw held for its lifetime and shared unchanged by every circuit that reads it, however many instants or distribution-net points that covers. Thermal and flicker fluctuation on a reference node is neither reused across instants nor shared between the circuits reading them, so it is indexed by access rather than by instance and belongs to the reading circuit's own model — the same deviation counted in both places would be counted twice.

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

Stated assumption: the taps are ideal high-impedance nodes (no load-dependent droop, no signal current), so the block's cost is purely static; the initial accuracy is a per-die constant relative to the nominal tap, and the sourced value is time-invariant between fabrications. Each tap is non-negative; a `0` V tap is valid and denotes a ground/rail reference (relative tolerance scales it to exactly `0`, so it stays stable and correct). Mode selection is quasi-static: switching the active mode is far slower than a read, so no per-read switching energy is modelled. Any departure belonging to the circuit that reads the tap — a comparator offset, a driver offset, a per-access thermal draw — is a property of that circuit and is modelled there, so the reference carries no second spread per reading site; counting it in both places would double the same physical deviation. The model is valid where the reference distribution load is negligible and the bias power is well-approximated as a constant.

TODO (domain author): the validity boundary of the load-free idealisation, and whether slow temperature drift within an inference must be modelled as a correlated term on the source.

## Validation

TODO — link validation evidence once written.

## References

TODO: cite the reference-accuracy and drift models.

---

- **Internals**: [voltage_reference internals](../../../internals/primitive/analog/voltage_reference.md)
- **Validation**: TODO — validation evidence not yet written
- **Configuration**: `VrefConfig`, `VrefPolicy` (see `api`)
