# NMOS Transistor

## Summary

`NMOS` is a pure electrical primitive: a continuous, three-terminal current-voltage model of an NMOS field-effect transistor, with temperature-scaled parameters and Pelgrom-law fabrication mismatch. In the device→circuit→architecture stack it sits at the bottom as a gate-drain-source primitive, exposing its I-V law and the three node partials $\partial I_{\mathrm{ds}}/\partial\{V_g,V_d,V_s\}$ to a consuming circuit. It owns the electrical law, the temperature scaling, and the mismatch sampling of its threshold voltage and transconductance factor; it does **not** own layout-dependent lumped parasitic capacitances, which belong to the circuit that places the transistor.

## Physical model

The transistor is modeled as a smooth, single-piece I-V surface valid across subthreshold and above-threshold operation, parameterized by a threshold voltage $V_{\mathrm{th}}$ and a transconductance factor $\beta$. The surface is an EKV-style symmetric formulation in the source- and drain-referred overdrive voltages, with the hard square-law corner replaced by a softplus so the law and its partials are continuous everywhere — required because the consuming solve needs the three node derivatives $\partial I_{\mathrm{ds}}/\partial\{V_g,V_d,V_s\}$ to assemble a Jacobian. The softplus / sigmoid smoothing scale is set by the subthreshold-swing factor $n$ and the thermal voltage $V_T = k_B T / q$, so the smoothing tracks temperature.

Both $\beta$ and $V_{\mathrm{th}}$ are temperature-scaled from their reference-temperature nominals, then perturbed by static per-cell fabrication mismatch sampled at fabricate time.

## Governing equations

**Temperature scaling.** With reference temperature $T_{\mathrm{ref}}$, the mobility follows a power law and the threshold a linear shift, giving the nominal transconductance factor and threshold:

$$\beta_{\mathrm{nom}} = \mu_0\left(\frac{T}{T_{\mathrm{ref}}}\right)^{-u_{\mathrm{te}}} C_{\mathrm{ox}}\,\frac{W}{L}, \qquad V_{\mathrm{th,nom}} = V_{\mathrm{th0}} + k_{t1}\left(\frac{T}{T_{\mathrm{ref}}}-1\right).$$

**I-V law.** With the source- and drain-referred overdrives $V_{\mathrm{ov,s}} = V_g - V_s - V_{\mathrm{th}}$ and $V_{\mathrm{ov,d}} = V_g - V_d - V_{\mathrm{th}}$, define the softplus-smoothed effective overdrives and their sigmoid derivatives at smoothing scale $\lambda = 1/(2 n V_T)$:

$$v_s = \operatorname{softplus}_\lambda(V_{\mathrm{ov,s}}), \quad v_d = \operatorname{softplus}_\lambda(V_{\mathrm{ov,d}}), \qquad \sigma_s = \operatorname{sigmoid}(\lambda V_{\mathrm{ov,s}}), \quad \sigma_d = \operatorname{sigmoid}(\lambda V_{\mathrm{ov,d}}),$$

where $\operatorname{softplus}_\lambda(u) = \lambda^{-1}\log(1+e^{\lambda u})$ and $\sigma = \mathrm{d}\,\operatorname{softplus}_\lambda/\mathrm{d}u$. The drain-source current is the difference of source- and drain-side square-law terms:

$$I_{\mathrm{ds}} = \tfrac{1}{2}\,\beta\left(v_s^2 - v_d^2\right),$$

with the three node partials in closed form:

$$\frac{\partial I_{\mathrm{ds}}}{\partial V_g} = \beta\left(v_s\sigma_s - v_d\sigma_d\right), \qquad \frac{\partial I_{\mathrm{ds}}}{\partial V_d} = \beta\,v_d\sigma_d \ge 0, \qquad \frac{\partial I_{\mathrm{ds}}}{\partial V_s} = -\beta\,v_s\sigma_s \le 0.$$

The sign convention is positive current for drain→source flow.

## Numerical method

N/A — the I-V surface and its three partials are evaluated in closed form; the transistor holds no solver. The consuming circuit assembles $I_{\mathrm{ds}}$ and the node partials into its operating-point Jacobian.

## Noise & non-idealities

Fabrication mismatch is Pelgrom-law area-scaled Gaussian noise sampled once per `fabricate()` call onto the per-instance threshold and transconductance maps. Each source is switched by a per-run policy flag (identity map when off):

- **$V_{\mathrm{th}}$ mismatch** (`A_vt_mismatch`, fabricate time) — additive Gaussian on $V_{\mathrm{th,nom}}$ with sigma $\sigma_{V_{\mathrm{th}}} = A_{V_{\mathrm{th}}}\cdot 10^{-3}/\sqrt{W L}$ (the $10^{-3}$ converts the mV-um matching coefficient to V).
- **$\beta$ mismatch** (`A_beta_mismatch`, fabricate time) — additive Gaussian on $\beta_{\mathrm{nom}}$ with relative sigma $\sigma_\beta/\beta = A_\beta/\sqrt{W L}$.

Both sigmas scale as $1/\sqrt{W L}$: larger devices match better. They are static device-to-device variation, not per-read noise.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `mu0__cm2_per_V_s` | low-field carrier mobility $\mu_0$ at $T_{\mathrm{ref}}$ | cm^2/V/s | Process |
| `c_ox__fF_per_um2` | gate-oxide capacitance per unit area $C_{\mathrm{ox}}$ | fF/um^2 | Process |
| `vth0__V` | nominal threshold $V_{\mathrm{th0}}$ at $T_{\mathrm{ref}}$ | V | Process |
| `n_factor` | subthreshold-swing factor $n$ ($>1$) | — | Process |
| `T_ref__K` | reference temperature $T_{\mathrm{ref}}$ | K | Process |
| `ute` | mobility temperature exponent $u_{\mathrm{te}}$ | — | Process |
| `kt1__V` | $V_{\mathrm{th}}$ temperature coefficient $k_{t1}$ | V | Process |
| `A_vt__mV_um` | Pelgrom $V_{\mathrm{th}}$ matching coefficient $A_{V_{\mathrm{th}}}$ | mV-um | Process |
| `A_beta_relative__um` | Pelgrom relative-$\beta$ matching coefficient $A_\beta$ | um | Process |
| `W__um` | channel width $W$ (init kwarg) | um | Design |
| `L__um` | channel length $L$ (init kwarg) | um | Design |

Provenance terms are defined in [parameter_provenance](../parameter_provenance.md). The thermal voltage $V_T = k_B T / q$ is derived from the constants $k_B$, $q$ (see [notation_conventions](../notation_conventions.md#physical-constants)). File-level schema: `api`.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $I_{\mathrm{ds}}$ | drain-source current (drain→source positive) | uA | `NMOSDCOP.ids__uA` |
| $V_g, V_d, V_s$ | gate, drain, source voltages (runtime inputs) | V | `vg__V`, `vd__V`, `vs__V` |
| $V_{\mathrm{ov,s}}, V_{\mathrm{ov,d}}$ | source-, drain-referred overdrive | V | — |
| $v_s, v_d$ | softplus-smoothed effective overdrives | V | `v_eff_s`, `v_eff_d` |
| $\sigma_s, \sigma_d$ | sigmoid derivatives of the softplus | — | `sigma_s`, `sigma_d` |
| $\beta$ | per-cell transconductance factor | uA/V^2 | `beta__uA_per_V2` |
| $V_{\mathrm{th}}$ | per-cell threshold voltage | V | `vth__V` |
| $\beta_{\mathrm{nom}}, V_{\mathrm{th,nom}}$ | temperature-scaled nominals | uA/V^2, V | `nominal_beta__uA_per_V2`, `nominal_vth__V` |
| $\lambda$ | softplus / sigmoid smoothing scale | 1/V | `_inv_smooth_scale__per_V` |
| $\mu_0$ | low-field mobility at $T_{\mathrm{ref}}$ | cm^2/V/s | `mu0__cm2_per_V_s` |
| $C_{\mathrm{ox}}$ | gate-oxide capacitance per area | fF/um^2 | `c_ox__fF_per_um2` |
| $n$ | subthreshold-swing factor | — | `n_factor` |
| $V_T$ | thermal voltage $k_B T / q$ | V | `thermal_voltage__V(T__K)` |
| $u_{\mathrm{te}}, k_{t1}$ | mobility exponent, $V_{\mathrm{th}}$ temperature coefficient | —, V | `ute`, `kt1__V` |
| $A_{V_{\mathrm{th}}}, A_\beta$ | Pelgrom matching coefficients | mV-um, um | `A_vt__mV_um`, `A_beta_relative__um` |
| $\sigma_{V_{\mathrm{th}}}, \sigma_\beta$ | mismatch sigmas | V, uA/V^2 | `sigma_vth__V`, `sigma_beta__uA_per_V2` |
| $W, L$ | channel width, length | um | `W__um`, `L__um` |
| $T, T_{\mathrm{ref}}$ | operating, reference temperature | K | `T__K`, `T_ref__K` |

## Assumptions, scope & validity

Stated assumptions of the current model:

- A single smooth EKV-softplus surface spans subthreshold and above-threshold operation; no separate region piecing.
- The square-law corner is softened by a softplus whose scale tracks $n V_T$; the model is exact only in the sharp-corner limit and smooths the transition otherwise.
- Mobility scales as a power law in temperature and $V_{\mathrm{th}}$ shifts linearly; higher-order temperature effects are not modeled.
- Mismatch is static (sampled at fabricate time) and Pelgrom area-scaled; no per-read electrical noise (e.g. flicker, thermal channel noise) is modeled at this level.
- Layout-dependent parasitic capacitances are out of scope and are held by the consuming circuit.

TODO (domain author): give the quantitative validity ranges — overdrive / drain-bias range over which the EKV-softplus surface matches the target device, body-effect treatment (the model has no explicit body terminal), and the temperature range of the scaling laws.

## Validation

TODO: link `validation/device` evidence — I-V and node-partial agreement against the analytic EKV reference, finite-difference checks of the three partials, and Pelgrom mismatch-statistics checks.

## References

TODO: cite the EKV transistor model and the Pelgrom mismatch law.

---

- **Internals**: [nmos internals](../../internals/device/nmos.md)
- **Validation**: TODO — `validation/device` (not yet written)
- **Configuration**: `api` (`NMOSConfig`, `NMOSPolicy`)
- **Decisions**: [ADR-0002 NMOS is a pure electrical primitive](../../about/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md)
