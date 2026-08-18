# MOSFET transistor

A continuous EKV-softplus current-voltage model of a three-terminal MOSFET: from the gate, drain, and source voltages it returns the drain-source current $I_{\mathrm{ds}}$ and its three node partials $\partial I_{\mathrm{ds}}/\partial\{V_g,V_d,V_s\}$ in closed form, with one model core serving both n-channel ($p=+1$) and p-channel ($p=-1$) devices. The threshold voltage and transconductance factor scale with temperature and carry Pelgrom fabrication mismatch.

## Physical model

The transistor is modeled as a smooth, single-piece I-V surface valid across subthreshold and above-threshold operation, parameterized by a signed threshold voltage $V_{\mathrm{th}}$, a positive transconductance magnitude $\beta$, and a channel polarity $p$ that selects n- versus p-channel. The surface is an EKV-style symmetric formulation in the polarity-scaled source- and drain-referred overdrive voltages, with the hard square-law corner replaced by a softplus so the law and its three node partials $\partial I_{\mathrm{ds}}/\partial\{V_g,V_d,V_s\}$ are continuous everywhere. The softplus / sigmoid smoothing scale is set by the subthreshold-swing factor $n$ and the thermal voltage $V_T = k_B T / q$, so the smoothing tracks temperature.

The threshold sign is set by the device's enhancement / depletion flavor, not by channel polarity; $\beta$ is always a positive magnitude, and the polarity factor $p$ enters the I-V law itself rather than being baked into either parameter. Both $\beta$ and $V_{\mathrm{th}}$ are temperature-scaled from their reference-temperature nominals, then perturbed by static per-cell fabrication mismatch sampled at fabricate time.

## Governing equations

**Temperature scaling.** With reference temperature $T_{\mathrm{ref}}$, the mobility follows the BSIM UTE power law and the threshold a linear shift, giving the nominal transconductance factor and threshold:

$$\beta_{\mathrm{nom}} = \mu_0\left(\frac{T}{T_{\mathrm{ref}}}\right)^{-u_{\mathrm{te}}} C_{\mathrm{ox}}\,\frac{W}{L}, \qquad V_{\mathrm{th,nom}} = V_{\mathrm{th0}} + k_{t1}\left(\frac{T}{T_{\mathrm{ref}}}-1\right).$$

The exponent $u_{\mathrm{te}}$ is the positive magnitude of the SPICE/BSIM `UTE` mobility temperature parameter (e.g. $1.5$); applied as $-u_{\mathrm{te}}$, mobility falls with temperature. $\beta_{\mathrm{nom}}$ is a positive magnitude for both polarities.

**I-V law.** With polarity $p\in\{+1,-1\}$ and the polarity-scaled source- and drain-referred overdrives $V_{\mathrm{ov,s}} = p\,(V_g - V_s - V_{\mathrm{th}})$ and $V_{\mathrm{ov,d}} = p\,(V_g - V_d - V_{\mathrm{th}})$, define the softplus-smoothed effective overdrives and their sigmoid derivatives at smoothing scale $\lambda = 1/(2 n V_T)$:

$$v_s = \operatorname{softplus}_\lambda(V_{\mathrm{ov,s}}), \quad v_d = \operatorname{softplus}_\lambda(V_{\mathrm{ov,d}}), \qquad \sigma_s = \operatorname{sigmoid}(\lambda V_{\mathrm{ov,s}}), \quad \sigma_d = \operatorname{sigmoid}(\lambda V_{\mathrm{ov,d}}),$$

where $\operatorname{softplus}_\lambda(u) = \lambda^{-1}\log(1+e^{\lambda u})$ and $\sigma = \mathrm{d}\,\operatorname{softplus}_\lambda/\mathrm{d}u$. The drain-source current carries a single polarity factor and is the difference of source- and drain-side square-law terms:

$$I_{\mathrm{ds}} = \tfrac{1}{2}\,p\,\beta\left(v_s^2 - v_d^2\right),$$

with the three node partials in closed form:

$$\frac{\partial I_{\mathrm{ds}}}{\partial V_g} = \beta\left(v_s\sigma_s - v_d\sigma_d\right), \qquad \frac{\partial I_{\mathrm{ds}}}{\partial V_d} = \beta\,v_d\sigma_d \ge 0, \qquad \frac{\partial I_{\mathrm{ds}}}{\partial V_s} = -\beta\,v_s\sigma_s \le 0.$$

The polarity factor squares out of the terminal partials: $p$ scales both the overdrives and the current prefactor, and $p^2 = 1$, so the partials carry no explicit polarity and the $\partial I/\partial V_d \ge 0$ / $\partial I/\partial V_s \le 0$ contract holds for both polarities. The sign convention is positive current for drain→source flow; an n-channel device in normal conduction carries $I_{\mathrm{ds}} > 0$, while a p-channel device in normal conduction carries $I_{\mathrm{ds}} < 0$.

The small-signal transconductance $g_m$ and output conductance $1/r_o$ follow directly from the node partials:

$$g_m = \frac{\partial I_{\mathrm{ds}}}{\partial V_{\mathrm{gs}}} = \frac{\partial I_{\mathrm{ds}}}{\partial V_g} = \beta\left(v_s\sigma_s - v_d\sigma_d\right), \qquad \frac{1}{r_o} = \frac{\partial I_{\mathrm{ds}}}{\partial V_{\mathrm{ds}}} = \frac{\partial I_{\mathrm{ds}}}{\partial V_d} = \beta\,v_d\sigma_d.$$

## Numerical method

N/A — the I-V surface and its three node partials are evaluated in closed form, with no iteration.

## Noise & non-idealities

Fabrication mismatch is Pelgrom-law area-scaled Gaussian noise on the per-instance threshold and transconductance maps.

- **$V_{\mathrm{th}}$ mismatch** (`A_vt_mismatch`, fabricate time) — additive Gaussian on $V_{\mathrm{th,nom}}$ with sigma $\sigma_{V_{\mathrm{th}}} = A_{V_{\mathrm{th}}}\cdot 10^{-3}/\sqrt{W L}$ (the $10^{-3}$ converts the mV-um matching coefficient to V).
- **$\beta$ mismatch** (`A_beta_mismatch`, fabricate time) — additive Gaussian on $\beta_{\mathrm{nom}}$ with relative sigma $\sigma_\beta/\beta = A_\beta/\sqrt{W L}$; since $\beta_{\mathrm{nom}}$ is a positive magnitude, $\sigma_\beta$ is positive directly.

$A_{V_{\mathrm{th}}}$ and $A_\beta$ are the standard Pelgrom area-matching coefficients: both sigma values scale as $1/\sqrt{W L}$, so larger devices match better; the shared area-scaled mismatch law is in [nonideality](../nonideality.md). They are static device-to-device variation, not per-read noise.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `mu0__cm2_per_V_s` | low-field carrier mobility $\mu_0$ at $T_{\mathrm{ref}}$ | cm^2/V/s | $> 0$ | Process |
| `c_ox__fF_per_um2` | gate-oxide capacitance per unit area $C_{\mathrm{ox}}$ | fF/um^2 | $> 0$ | Process |
| `vth0__V` | signed nominal threshold $V_{\mathrm{th0}}$ at $T_{\mathrm{ref}}$ | V | — | Process |
| `n_factor` | subthreshold-swing factor $n$ | — | $> 1$ | Process |
| `T_nom__K` | reference temperature $T_{\mathrm{ref}}$ | K | $> 0$ | Process |
| `ute` | mobility temperature exponent $u_{\mathrm{te}}$ | — | $\ge 0$ | Process |
| `kt1__V` | $V_{\mathrm{th}}$ temperature coefficient $k_{t1}$ | V | — | Process |
| `A_vt__mV_um` | Pelgrom $V_{\mathrm{th}}$ matching coefficient $A_{V_{\mathrm{th}}}$ | mV-um | $\ge 0$ | Process |
| `A_beta_relative__um` | Pelgrom relative-$\beta$ matching coefficient $A_\beta$ | um | $\ge 0$ | Process |
| `W__um` | channel width $W$ (init kwarg) | um | $> 0$ | Design |
| `L__um` | channel length $L$ (init kwarg) | um | $> 0$ | Design |

The channel polarity $p$ is fixed by device type ($+1$ n-channel, $-1$ p-channel), not a tunable parameter, so it is not listed above. Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md). The thermal voltage $V_T = k_B T / q$ is derived from the constants $k_B$, $q$ (see [notation_conventions](../../../conventions/notation_conventions.md#physical-constants)). File-level schema: `api`.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $I_{\mathrm{ds}}$ | drain-source current (drain→source positive) | uA | `MosfetDcop.ids__uA` |
| $\partial I_{\mathrm{ds}}/\partial V_g$ | gate transconductance $g_m$ | uS | `MosfetDcop.did_dvg__uS` |
| $\partial I_{\mathrm{ds}}/\partial V_d$ | drain conductance $1/r_o$ ($\ge 0$) | uS | `MosfetDcop.did_dvd__uS` |
| $\partial I_{\mathrm{ds}}/\partial V_s$ | source conductance ($\le 0$) | uS | `MosfetDcop.did_dvs__uS` |
| $p$ | channel polarity (+1 n-channel, -1 p-channel) | — | `polarity` |
| $V_g, V_d, V_s$ | gate, drain, source voltages (runtime inputs) | V | `vg__V`, `vd__V`, `vs__V` |
| $V_{\mathrm{ov,s}}, V_{\mathrm{ov,d}}$ | polarity-scaled source-, drain-referred overdrive | V | — |
| $v_s, v_d$ | softplus-smoothed effective overdrives | V | `v_eff_s`, `v_eff_d` |
| $\sigma_s, \sigma_d$ | sigmoid derivatives of the softplus | — | `sigma_s`, `sigma_d` |
| $\beta$ | per-cell transconductance-factor magnitude | uA/V^2 | `MosfetSnap.beta__uA_per_V2` |
| $V_{\mathrm{th}}$ | per-cell signed threshold voltage | V | `MosfetSnap.vth__V` |
| $\beta_{\mathrm{nom}}, V_{\mathrm{th,nom}}$ | temperature-scaled internal nominals | uA/V^2, V | `_nominal_beta__uA_per_V2`, `_nominal_vth__V` |
| $\lambda$ | softplus / sigmoid smoothing scale | 1/V | `_inv_smooth_scale__per_V` |
| $\mu_0$ | low-field mobility at $T_{\mathrm{ref}}$ | cm^2/V/s | `mu0__cm2_per_V_s` |
| $C_{\mathrm{ox}}$ | gate-oxide capacitance per area | fF/um^2 | `c_ox__fF_per_um2` |
| $n$ | subthreshold-swing factor | — | `n_factor` |
| $V_T$ | thermal voltage $k_B T / q$ | V | `thermal_voltage__V(T__K)` |
| $u_{\mathrm{te}}, k_{t1}$ | mobility exponent, $V_{\mathrm{th}}$ temperature coefficient | —, V | `ute`, `kt1__V` |
| $A_{V_{\mathrm{th}}}, A_\beta$ | Pelgrom matching coefficients | mV-um, um | `A_vt__mV_um`, `A_beta_relative__um` |
| $\sigma_{V_{\mathrm{th}}}, \sigma_\beta$ | mismatch sigma values | V, uA/V^2 | `sigma_vth__V`, `sigma_beta__uA_per_V2` |
| $W, L$ | channel width, length | um | `W__um`, `L__um` |
| $T, T_{\mathrm{ref}}$ | operating, reference temperature | K | `T__K`, `T_nom__K` |

## Assumptions, scope & validity

- A single smooth EKV-softplus surface spans subthreshold and above-threshold operation; no separate region piecing.
- The square-law corner is softened by a softplus whose scale tracks $n V_T$; the model is exact only in the sharp-corner limit and smooths the transition otherwise.
- Polarity is a discrete $\pm 1$ selector distinguishing the two channel types; one model core serves both, with the threshold sign and any n/p process asymmetry carried by each device's own $V_{\mathrm{th0}}$ and process values rather than modeled intrinsically.
- Mobility scales as a power law in temperature and $V_{\mathrm{th}}$ shifts linearly; higher-order temperature effects are not modeled.
- Mismatch is static (sampled at fabricate time) and Pelgrom area-scaled; no per-read electrical noise (e.g. flicker, thermal channel noise) is modeled at this level.
- Layout-dependent parasitic capacitances are out of scope.

TODO (domain author): give the quantitative validity ranges — overdrive / drain-bias range over which the EKV-softplus surface matches the target device for each polarity, body-effect treatment (the model has no explicit body terminal), and the temperature range of the scaling laws.

## Validation

TODO: link `validation/device` evidence — I-V and node-partial agreement against the analytic EKV reference for both polarities, finite-difference checks of the three partials, and Pelgrom mismatch-statistics checks.

## References

TODO: cite the EKV transistor model and the Pelgrom mismatch law.
