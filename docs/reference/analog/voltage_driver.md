# Voltage driver

## Physical model

The clamp is modelled as a Thevenin equivalent: a reference voltage source $V_{\mathrm{ref}}$ (the open-circuit clamp voltage) in series with a constant output resistance $R_{\mathrm{out}}$. The source sets the held voltage at zero current; the series resistance is the lumped output impedance through which the port current flows, so the clamp voltage droops linearly with the current the clamp sources or sinks. The model lumps the small-signal output impedance of a bounded-impedance drive or sense node, over the operating range, into the single constant $R_{\mathrm{out}}$; setting $R_{\mathrm{out}} = 0$ recovers the ideal voltage source whose held voltage never droops. The reference voltage $V_{\mathrm{ref}}$ is a supplied input, not an internal constant of the clamp. Two modelled non-idealities perturb that reference: a systematic per-instance offset and thermal noise.

A Thevenin source dissipates no signal power of its own: the conduction power that holds the clamp under load is dissipated in the supply rail, not inside the clamp, so this model does not tally it. The block's own dissipation is its static power alone (leakage), including any internal amplifier or bias network.

## Governing equations

The clamp transfer function is the Thevenin map

$$V_{\mathrm{clamp}} = V_{\mathrm{ref}} - I_{\mathrm{port}} \, R_{\mathrm{out}},$$

where $V_{\mathrm{ref}}$ is the reference / zero-current clamp voltage (after the offset and thermal perturbations) and $R_{\mathrm{out}}$ is the series output resistance. The small-signal output resistance is the constant slope

$$\frac{\partial V_{\mathrm{clamp}}}{\partial I_{\mathrm{port}}} = -R_{\mathrm{out}},$$

a resistance in MOhm; in the consistent unit set $\mathrm{uA} \times \mathrm{MOhm} = \mathrm{V}$. At $R_{\mathrm{out}} = 0$ the map collapses to the constant $V_{\mathrm{clamp}} = V_{\mathrm{ref}}$ with a zero derivative — the ideal constant-voltage source. The response is monotone, non-increasing for $R_{\mathrm{out}} \ge 0$.

## Numerical method

N/A — closed-form affine map and its constant derivative; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| reference offset | systematic per-instance mismatch of the reference source | additive zero-mean Gaussian on $V_{\mathrm{ref}}$, state-independent (a single constant sigma) | offset sigma |
| reference thermal noise | thermal fluctuation on the reference node | additive zero-mean Gaussian on $V_{\mathrm{ref}}$, state-independent (a single constant sigma) | thermal sigma |

TODO (domain author): give each sigma's physical derivation and citation, and confirm whether the thermal term carries any temperature scaling.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `r_out__MOhm` ($R_{\mathrm{out}}$) | series output resistance / constant clamp slope | MOhm | $\geq 0$ | Design |
| `offset_sigma__V` | Gaussian sigma of the systematic reference offset | V | $\geq 0$ | Measured |
| `thermal_sigma__V` | Gaussian sigma of the reference thermal noise | V | $\geq 0$ | Measured |
| leakage / area | static PPA / spec fields (leakage carries all static power, incl. internal amplifier / bias) | uW, um^2 | $\geq 0$ | Design |

The reference $V_{\mathrm{ref}}$ is a supplied runtime input, not a config parameter of the driver. Provenance terms are defined in [module_parameter](../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{clamp}}$ | clamp voltage | V | `solve_clamp` return |
| $V_{\mathrm{ref}}$ | reference / zero-current clamp voltage (supplied input) | V | `snapshot(v_ref__V=...)`, `VoltageDriverSnap.v_ref__V` |
| $R_{\mathrm{out}}$ | series output resistance | MOhm | `r_out__MOhm` |
| $I_{\mathrm{port}}$ | port current | uA | `i_port__uA` |

## Assumptions, scope & validity

Stated assumption: the clamp is a Thevenin source — a constant reference voltage behind a constant series resistance — so its output impedance is a single value independent of operating point. The model is valid where the lumped $R_{\mathrm{out}}$ captures the node's small-signal output impedance over the operating range (no headroom / compliance limit, no impedance that varies with current).

TODO (domain author): the validity boundary of the constant-$R_{\mathrm{out}}$ idealisation (the current range over which a real clamp's output impedance stays linear), and whether finite slew / settling within the read window is neglected.

## Validation

TODO — link validation evidence once written: that $R_{\mathrm{out}} \to 0$ recovers the ideal constant-voltage source, and that the affine clamp reproduces the specified droop.

## References

TODO: cite the Thevenin-equivalent clamp model.

---

- **Internals**: [voltage_driver internals](../../internals/analog/voltage_driver.md)
- **Validation**: TODO — validation evidence not yet written
- **Configuration**: `VoltageDriverConfig`, `VoltageDriverPolicy` (see `api`)
