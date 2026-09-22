# Voltage driver

## Physical model

The clamp is a Thevenin source with open-circuit voltage $V_{\mathrm{open}}$ and constant series resistance $R_{\mathrm{out}}$. Its supplied nominal reference $V_{\mathrm{ref}}$ receives an additive perturbation $V_{\mathrm{pert}}$ from the clamp's offset and thermal noise.

Each settled port operation incurs a fixed interface-node switching energy $E_{\mathrm{op}}$, representing one full $C V^2$ precharge cycle at the design point. Static power includes leakage and any internal amplifier or bias network.

## Governing equations

The transfer is

$$V_{\mathrm{open}} = V_{\mathrm{ref}} + V_{\mathrm{pert}}, \qquad
V_{\mathrm{port}} = V_{\mathrm{open}} - I_{\mathrm{port}} \, R_{\mathrm{out}},$$

with constant small-signal slope

$$\frac{\partial V_{\mathrm{port}}}{\partial I_{\mathrm{port}}} = -R_{\mathrm{out}},$$

in MOhm, since $\mathrm{uA} \times \mathrm{MOhm} = \mathrm{V}$. The response is non-increasing for $R_{\mathrm{out}} \ge 0$. At zero resistance, $V_{\mathrm{port}} = V_{\mathrm{ref}} + V_{\mathrm{pert}}$; disabling both perturbations recovers the ideal reference voltage.

Energy per port operation is

$$E = E_{\mathrm{op}},$$

so holding $n$ ports for one access costs $n E_{\mathrm{op}}$.

## Numerical method

N/A — closed-form affine map and its constant derivative; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
| --- | --- | --- | --- |
| clamp offset | per-instance mismatch fixed at fabrication | additive zero-mean Gaussian in $V_{\mathrm{pert}}$, constant sigma | offset sigma |
| clamp thermal noise | node fluctuation sampled independently per access | additive zero-mean Gaussian in $V_{\mathrm{pert}}$, constant sigma | thermal sigma |

TODO (domain author): give each sigma's physical derivation and citation, and confirm whether the thermal term carries any temperature scaling.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
| --- | --- | --- | --- | --- |
| `r_out__MOhm` ($R_{\mathrm{out}}$) | series output resistance | MOhm | $\geq 0$ | Design |
| `offset_sigma__V` | Gaussian sigma of the systematic clamp offset | V | $\geq 0$ | Measured |
| `thermal_sigma__V` | Gaussian sigma of the clamp thermal noise | V | $\geq 0$ | Measured |
| `energy_per_op__fJ` ($E_{\mathrm{op}}$) | interface-node switching energy per port operation | fJ | $\geq 0$ | Design |
| leakage / area | total static power and silicon area | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $V_{\mathrm{port}}$ | port voltage | V | `v_port__V` |
| $V_{\mathrm{ref}}$ | nominal reference voltage (supplied input) | V | `v_ref__V` |
| $V_{\mathrm{open}}$ | zero-load output including offset and thermal noise | V | `v_open__V` |
| $V_{\mathrm{pert}}$ | additive offset plus thermal draw | V | — |
| $R_{\mathrm{out}}$ | series output resistance | MOhm | `r_out__MOhm` |
| $\partial V_{\mathrm{port}}/\partial I_{\mathrm{port}}$ | small-signal port-voltage slope | MOhm | `dvport_di__MOhm` |
| $I_{\mathrm{port}}$ | port current | uA | `i_port__uA` |
| $E_{\mathrm{op}}$ | interface-node switching energy per port operation | fJ | `energy_per_op__fJ` |

## Assumptions, scope & validity

The model applies where constant $R_{\mathrm{out}}$ represents the node's small-signal output impedance. It excludes headroom and compliance limits and current-dependent impedance.

The energy $E_{\mathrm{op}}$ is independent of voltage, current, and $R_{\mathrm{out}}$. The model excludes state-dependent interface swings and duration-dependent branch dissipation, including $I_{\mathrm{port}}^{2}R_{\mathrm{out}}$ loss.

TODO (domain author): the validity boundary of the constant-$R_{\mathrm{out}}$ idealisation (the current range over which a real clamp's output impedance stays linear), and whether finite slew / settling within the read window is neglected.

## Validation

TODO — link validation evidence once written: that $R_{\mathrm{out}} \to 0$ recovers the ideal constant-voltage source, and that the affine clamp reproduces the specified droop.

## References

TODO: cite the Thevenin-equivalent clamp model.
