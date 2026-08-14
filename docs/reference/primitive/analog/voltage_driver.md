# Voltage driver

## Physical model

The clamp is modelled as a Thevenin equivalent: a reference voltage source $V_{\mathrm{ref}}$ (the open-circuit clamp voltage) in series with a constant output resistance $R_{\mathrm{out}}$. The source sets the held voltage at zero current; the series resistance is the lumped output impedance through which the port current flows, so the clamp voltage droops linearly with the current the clamp sources or sinks. The model lumps the small-signal output impedance of a bounded-impedance drive or sense node, over the operating range, into the single constant $R_{\mathrm{out}}$; setting $R_{\mathrm{out}} = 0$ recovers the ideal voltage source whose held voltage never droops. The reference voltage $V_{\mathrm{ref}}$ is a supplied input, not an internal constant of the clamp, and it is the nominal level: the two modelled non-idealities — a systematic per-instance offset and thermal noise — are the clamp's own and are carried as a separate additive perturbation $V_{\mathrm{pert}}$ on top of it, so the ideal level the clamp was asked to hold stays legible beside the level it actually holds.

The clamp's dissipation splits by time base. Delivering the clamp once at a settled port state cycles the interface node, a fixed switching quantum $E_{\mathrm{op}}$ — one full $C V^2$ interface-node precharge cycle at the modelled design point — that the model tallies per port operation; a physical zero is a legitimate value for it, and the quantum is flat, independent of the port state it is delivered at. The timed conduction that holds the clamp under load is the other term: the rail-to-GND branch sustaining the port current dissipates $I_{\mathrm{port}}^{2} R_{\mathrm{out}}$ inside the lumped series resistance whenever $R_{\mathrm{out}} > 0$, plus whatever the rest of that branch drops. That term carries the conduction window of the access, which the clamp does not own — the clamp sees a settled operating point and no duration — so the whole branch draw is accounted where the conduction window lives, and the clamp itself splits none of it out. Static power (leakage, including any internal amplifier or bias network) is the remaining term.

## Governing equations

The clamp transfer function is the Thevenin map

$$V_{\mathrm{clamp}} = V_{\mathrm{ref}} + V_{\mathrm{pert}} - I_{\mathrm{port}} \, R_{\mathrm{out}},$$

where $V_{\mathrm{ref}}$ is the nominal reference / zero-current clamp voltage, $V_{\mathrm{pert}}$ the clamp's own additive perturbation, and $R_{\mathrm{out}}$ the series output resistance. The small-signal output resistance is the constant slope

$$\frac{\partial V_{\mathrm{clamp}}}{\partial I_{\mathrm{port}}} = -R_{\mathrm{out}},$$

a resistance in MOhm; in the consistent unit set $\mathrm{uA} \times \mathrm{MOhm} = \mathrm{V}$. At $R_{\mathrm{out}} = 0$ the map collapses to the constant $V_{\mathrm{clamp}} = V_{\mathrm{ref}} + V_{\mathrm{pert}}$ with a zero derivative — the ideal constant-voltage source, which holds exactly $V_{\mathrm{ref}}$ once both perturbations are off. The response is monotone, non-increasing for $R_{\mathrm{out}} \ge 0$.

The energy the clamp accounts for one delivered access is the interface quantum alone,

$$E = E_{\mathrm{op}},$$

per port operation of the settled pair $(I_{\mathrm{port}}, V_{\mathrm{clamp}})$ — one operation per clamped port position, so a clamp holding $n$ ports over one access accounts $n E_{\mathrm{op}}$.

## Numerical method

N/A — closed-form affine map and its constant derivative; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| clamp offset | systematic per-instance mismatch of the clamp, fixed at fabrication | additive zero-mean Gaussian in $V_{\mathrm{pert}}$, state-independent (a single constant sigma) | offset sigma |
| clamp thermal noise | thermal fluctuation at the clamp node, resampled per access | additive zero-mean Gaussian in $V_{\mathrm{pert}}$, state-independent (a single constant sigma) | thermal sigma |

Both are the clamp's own: a reference source holds one static identity shared by every circuit reading it, so the fluctuation that differs from one access to the next is drawn here, at the access positions this clamp covers.

TODO (domain author): give each sigma's physical derivation and citation, and confirm whether the thermal term carries any temperature scaling.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `r_out__MOhm` ($R_{\mathrm{out}}$) | series output resistance, whose negative is the constant clamp slope | MOhm | $\geq 0$ | Design |
| `offset_sigma__V` | Gaussian sigma of the systematic clamp offset | V | $\geq 0$ | Measured |
| `thermal_sigma__V` | Gaussian sigma of the clamp thermal noise | V | $\geq 0$ | Measured |
| `energy_per_op__fJ` ($E_{\mathrm{op}}$) | interface-node switching energy per port operation | fJ | $\geq 0$ | Design |
| leakage / area | static PPA / spec fields (leakage carries all static power, incl. internal amplifier / bias) | uW, um^2 | $\geq 0$ | Design |

The reference $V_{\mathrm{ref}}$ is a supplied runtime input, not a config parameter of the driver. Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{clamp}}$ | clamp voltage | V | `v_clamp__V` |
| $V_{\mathrm{ref}}$ | nominal reference / zero-current clamp voltage (supplied input) | V | `v_ref__V` |
| $V_{\mathrm{pert}}$ | clamp's own additive perturbation (offset plus thermal draw) | V | `v_perturb__V` |
| $R_{\mathrm{out}}$ | series output resistance | MOhm | `r_out__MOhm` |
| $\partial V_{\mathrm{clamp}}/\partial I_{\mathrm{port}}$ | small-signal clamp slope | MOhm | `dvclamp_di__MOhm` |
| $I_{\mathrm{port}}$ | port current | uA | `i_port__uA` |
| $E_{\mathrm{op}}$ | interface-node switching energy per port operation | fJ | `energy_per_op__fJ` |

## Assumptions, scope & validity

Stated assumption: the clamp is a Thevenin source — a constant reference voltage behind a constant series resistance — so its output impedance is a single value independent of operating point. The model is valid where the lumped $R_{\mathrm{out}}$ captures the node's small-signal output impedance over the operating range (no headroom / compliance limit, no impedance that varies with current).

The energy accounting assumes a flat interface quantum: $E_{\mathrm{op}}$ is independent of the delivered voltage, of the port current, and of $R_{\mathrm{out}}$, so a design whose interface-node swing tracks the port state is outside this model. It also assumes the clamp is not the owner of the conduction window, so the timed branch draw — the $I_{\mathrm{port}}^{2} R_{\mathrm{out}}$ share included — is accounted once by whoever owns that window; a deployment where nothing owns it leaves that term uncounted.

TODO (domain author): the validity boundary of the constant-$R_{\mathrm{out}}$ idealisation (the current range over which a real clamp's output impedance stays linear), and whether finite slew / settling within the read window is neglected.

## Validation

TODO — link validation evidence once written: that $R_{\mathrm{out}} \to 0$ recovers the ideal constant-voltage source, and that the affine clamp reproduces the specified droop.

## References

TODO: cite the Thevenin-equivalent clamp model.

---

- **Internals**: [voltage_driver internals](../../../internals/primitive/analog/voltage_driver.md)
- **Validation**: TODO — validation evidence not yet written
- **Configuration**: `VoltageDriverConfig`, `VoltageDriverPolicy` (see `api`)
