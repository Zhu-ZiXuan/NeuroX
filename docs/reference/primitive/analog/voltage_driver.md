# Voltage driver

## Physical model

The clamp is modelled as a Thevenin equivalent: a reference voltage source $V_{\mathrm{ref}}$ (the open-circuit port voltage) in series with a constant output resistance $R_{\mathrm{out}}$. The source sets the held voltage at zero current; the series resistance is the lumped output impedance through which the port current flows, so the port voltage droops linearly with the current the clamp sources or sinks. The model lumps the small-signal output impedance of a bounded-impedance drive or sense node, over the operating range, into the single constant $R_{\mathrm{out}}$; setting $R_{\mathrm{out}} = 0$ recovers the ideal voltage source whose held voltage never droops. The reference voltage $V_{\mathrm{ref}}$ is a supplied input, not an internal constant of the clamp, and it is the nominal level: the two modelled non-idealities — a systematic per-instance offset and thermal noise — are the clamp's own and are carried as a separate additive perturbation $V_{\mathrm{pert}}$ on top of it.

Delivering the clamp once at a settled port state cycles the interface node, a fixed switching quantum $E_{\mathrm{op}}$ — one full $C V^2$ interface-node precharge cycle at the modeled design point — that is flat and independent of the port state. The model receives no conduction duration and therefore excludes duration-dependent branch dissipation, including the $I_{\mathrm{port}}^{2}R_{\mathrm{out}}$ loss in the lumped series resistance. Static power includes leakage and any internal amplifier or bias network.

## Governing equations

The clamp transfer function is the Thevenin map

$$V_{\mathrm{port}} = V_{\mathrm{ref}} + V_{\mathrm{pert}} - I_{\mathrm{port}} \, R_{\mathrm{out}},$$

where $V_{\mathrm{ref}}$ is the nominal reference / zero-current port voltage, $V_{\mathrm{pert}}$ the clamp's own additive perturbation, and $R_{\mathrm{out}}$ the series output resistance. The small-signal output resistance is the constant slope

$$\frac{\partial V_{\mathrm{port}}}{\partial I_{\mathrm{port}}} = -R_{\mathrm{out}},$$

a resistance in MOhm; in the consistent unit set $\mathrm{uA} \times \mathrm{MOhm} = \mathrm{V}$. At $R_{\mathrm{out}} = 0$ the map collapses to the constant $V_{\mathrm{port}} = V_{\mathrm{ref}} + V_{\mathrm{pert}}$ with a zero derivative — the ideal constant-voltage source, which holds exactly $V_{\mathrm{ref}}$ once both perturbations are off. The response is monotone, non-increasing for $R_{\mathrm{out}} \ge 0$.

The energy the clamp accounts for one delivered access is the interface quantum alone,

$$E = E_{\mathrm{op}},$$

per port operation of the settled pair $(I_{\mathrm{port}}, V_{\mathrm{port}})$ — one operation per clamped port position, so a clamp holding $n$ ports over one access accounts $n E_{\mathrm{op}}$.

## Numerical method

N/A — closed-form affine map and its constant derivative; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
| --- | --- | --- | --- |
| clamp offset | systematic per-instance mismatch of the clamp, fixed at fabrication | additive zero-mean Gaussian in $V_{\mathrm{pert}}$, state-independent (a single constant sigma) | offset sigma |
| clamp thermal noise | thermal fluctuation at the clamp node, resampled per access | additive zero-mean Gaussian in $V_{\mathrm{pert}}$, state-independent (a single constant sigma) | thermal sigma |

The offset is fixed per fabricated instance; thermal noise is drawn independently per access.

TODO (domain author): give each sigma's physical derivation and citation, and confirm whether the thermal term carries any temperature scaling.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
| --- | --- | --- | --- | --- |
| `r_out__MOhm` ($R_{\mathrm{out}}$) | series output resistance, whose negative is the constant port-voltage slope | MOhm | $\geq 0$ | Design |
| `offset_sigma__V` | Gaussian sigma of the systematic clamp offset | V | $\geq 0$ | Measured |
| `thermal_sigma__V` | Gaussian sigma of the clamp thermal noise | V | $\geq 0$ | Measured |
| `energy_per_op__fJ` ($E_{\mathrm{op}}$) | interface-node switching energy per port operation | fJ | $\geq 0$ | Design |
| leakage / area | static PPA / spec fields (leakage carries all static power, incl. internal amplifier / bias) | uW, um^2 | $\geq 0$ | Design |

The reference $V_{\mathrm{ref}}$ is a supplied runtime input, not a config parameter of the driver. Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $V_{\mathrm{port}}$ | port voltage | V | `v_port__V` |
| $V_{\mathrm{ref}}$ | nominal reference / zero-current port voltage (supplied input) | V | `v_ref__V` |
| $V_{\mathrm{pert}}$ | clamp's own additive perturbation (offset plus thermal draw) | V | `v_perturb__V` |
| $R_{\mathrm{out}}$ | series output resistance | MOhm | `r_out__MOhm` |
| $\partial V_{\mathrm{port}}/\partial I_{\mathrm{port}}$ | small-signal port-voltage slope | MOhm | `dvport_di__MOhm` |
| $I_{\mathrm{port}}$ | port current | uA | `i_port__uA` |
| $E_{\mathrm{op}}$ | interface-node switching energy per port operation | fJ | `energy_per_op__fJ` |

## Assumptions, scope & validity

Stated assumption: the clamp is a Thevenin source — a constant reference voltage behind a constant series resistance — so its output impedance is a single value independent of operating point. The model is valid where the lumped $R_{\mathrm{out}}$ captures the node's small-signal output impedance over the operating range (no headroom / compliance limit, no impedance that varies with current).

The energy accounting assumes a flat interface quantum: $E_{\mathrm{op}}$ is independent of the delivered voltage, port current, and $R_{\mathrm{out}}$, so a design whose interface-node swing tracks the port state is outside this model. Duration-dependent branch dissipation, including the $I_{\mathrm{port}}^{2} R_{\mathrm{out}}$ term, is outside the clamp cost model.

TODO (domain author): the validity boundary of the constant-$R_{\mathrm{out}}$ idealisation (the current range over which a real clamp's output impedance stays linear), and whether finite slew / settling within the read window is neglected.

## Validation

TODO — link validation evidence once written: that $R_{\mathrm{out}} \to 0$ recovers the ideal constant-voltage source, and that the affine clamp reproduces the specified droop.

## References

TODO: cite the Thevenin-equivalent clamp model.
