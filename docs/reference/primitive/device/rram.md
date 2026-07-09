# RRAM cell

A two-terminal resistive-memory cell whose single state variable is the programmed conductance $G$. A read draws current from the stored $G$ at the applied terminal voltage $V$ under a symmetric hyperbolic-sine I-V law, yielding both the current and the local differential conductance; a write drives $G$ toward a target $G_{\mathrm t}$ after an elapsed retention time $t$, subject to program-time and read-time non-idealities.

## Physical model

The device is abstracted as a programmable conductor with a single state variable, the per-cell conductance $G$, confined to a working range $[G_{\min}, G_{\max}]$. $G_{\min}$ is an intrinsic floor of the device; $G_{\max}$ is a design ceiling set by external current limiting during programming, not an intrinsic device property. Two distinct operations act on the stored $G$:

- a **write** that drives the stored conductance toward a target value, subject to programming variation, conductance drift over the elapsed retention time, and stuck-at faults;
- a **read** that draws current from the present stored conductance under telegraph and thermal read noise.

The conduction itself is taken as instantaneous and quasi-static: a read returns the DC current at the applied terminal voltage with no transient switching dynamics within the read. The per-cell electrode parasitic capacitances ($C_{\mathrm{top}}$ on the BL side, $C_{\mathrm{bot}}$ on the internal-node side) bear on the cell's dynamic energy but do not enter the conduction law.

## Governing equations

**Read I-V law.** The conduction is a symmetric hyperbolic-sine law parameterized by the nonlinearity factor $\alpha$, reducing to an ohmic law in the linear limit $\alpha \to 0$:

$$I_{\mathrm R}(V) = \frac{G}{\alpha}\,\sinh(\alpha V), \qquad \frac{\partial I_{\mathrm R}}{\partial V} = G\,\cosh(\alpha V),$$

with the ohmic limit

$$I_{\mathrm R}(V) = G\,V, \qquad \frac{\partial I_{\mathrm R}}{\partial V} = G \quad (\alpha = 0).$$

**Programming write.** A write to a target conductance $G_{\mathrm t}$ after elapsed time $t$ is the composition, in order, of a working-range clamp, state-dependent programming variation, a power-law drift gain, a stuck-at replacement, and a final clamp:

$$G \leftarrow \operatorname{clamp}\!\Big(\operatorname{stuck}\big(d(t)\cdot\Gamma(\operatorname{clamp}(G_{\mathrm t},\,G_{\min},\,G_{\max}))\big),\ G_{\min},\ G_{\max}\Big),$$

where $\Gamma(\cdot)$ is the state-dependent programming-variation map (§Noise), $\operatorname{stuck}(\cdot)$ the stuck-at map, and the drift gain is a power law applied only once the elapsed time exceeds the reference time $t_0$:

$$d(t) = \left(\frac{t}{t_0}\right)^{-\nu} \quad (t > t_0,\ \nu > 0), \qquad d(t) = 1 \quad \text{otherwise}.$$

## Numerical method

N/A — the read I-V and its derivative are evaluated in closed form; the cell holds no solver of its own.

## Noise & non-idealities

The statistical forms below are the device's own non-ideality sources; the shared parameterization convention is in [nonideality](../nonideality.md).

- **Programming variation** (program time) — a multiplicative Gamma perturbation, normalized to unit mean, whose shape parameter $k$ depends on the normalized conductance state $\hat G = (G - G_{\mathrm{lo}})/(G_{\mathrm{hi}} - G_{\mathrm{lo}})$: $k(\hat G) = \max(k_{\mathrm{slope}}\,\hat G + k_{\mathrm{int}},\,0.1)$ at fixed scale $\theta$. The normalization bounds $G_{\mathrm{lo}}, G_{\mathrm{hi}}$ belong to the programming-variation model and are independent of the device working-range bounds $G_{\min}, G_{\max}$. The applied gain is $\gamma/\mathbb{E}[\gamma]$ with $\gamma \sim \operatorname{Gamma}(k,\theta)$, so the perturbation preserves the mean conductance and only injects state-dependent spread.
- **Stuck-at fault** (program time) — each cell is independently forced to $G_{\min}$ with probability $p_{\min}$ or to $G_{\max}$ with probability $p_{\max}$ (requiring $p_{\min}+p_{\max}<1$), else left unchanged.
- **Conductance drift** (program time) — the power-law gain $d(t)$ above, applied when $\nu>0$ and $t>t_0$.
- **Telegraph read noise** (read time) — random telegraph noise sampled on each read: a cell is in the active state with probability $p_{\mathrm{high}}$, and when active receives an additive perturbation of random sign and Gaussian-distributed amplitude (mean $\mu_a$, std $\sigma_a$).
- **Thermal read noise** (read time) — additive zero-mean Gaussian noise of std $\sigma_{\mathrm{th}}$ on the read conductance.

After the read-time sources, the read conductance is re-clamped to $[G_{\min}, G_{\max}]$.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `g_min__uS` | minimum programmable conductance $G_{\min}$ | uS | $\ge 0$ | Measured |
| `g_max__uS` | maximum programmable conductance $G_{\max}$ (init kwarg, design ceiling) | uS | $> G_{\min}$ | Design |
| `nonlinearity_alpha` | hyperbolic-sine I-V factor $\alpha$ | 1/V | $\ge 0$ | Measured |
| `drift_decay_rate` | power-law drift exponent $\nu$ | — | $\ge 0$ | Measured |
| `drift_t0` | reference drift time $t_0$ | s | $> 0$ | Measured |
| `c_top__fF` | top-electrode (BL-side) parasitic capacitance per cell $C_{\mathrm{top}}$ | fF | $\ge 0$ | Process |
| `c_bot__fF` | bottom-electrode (internal-node-side) parasitic capacitance per cell $C_{\mathrm{bot}}$ | fF | $\ge 0$ | Process |
| `read_thermal__uS` | thermal read-noise sigma $\sigma_{\mathrm{th}}$ | uS | $\ge 0$ | Measured |
| `prog_gamma` ($k_{\mathrm{slope}}, k_{\mathrm{int}}, \theta$, norm range) | state-dependent programming-variation parameters | — | $\theta > 0$; $k_{\mathrm{int}} > 0$; $G_{\mathrm{hi}} > G_{\mathrm{lo}}$ | Measured |
| `read_telegraph` ($\mu_a, \sigma_a, p_{\mathrm{high}}$) | telegraph read-noise parameters | uS, uS, — | $\sigma_a \ge 0$; $0 \le p_{\mathrm{high}} \le 1$ | Measured |
| `stuck_at` ($p_{\min}, p_{\max}$) | stuck-at fault probabilities | — | $p_{\min}, p_{\max} \ge 0$; $p_{\min} + p_{\max} < 1$ | Measured |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md). How to obtain values for a new chip: `guides/calibration`; file-level schema: `api`.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $G$ | per-cell programmed conductance | uS | `g__uS` |
| $G_{\min}, G_{\max}$ | working-range bounds | uS | `g_min__uS`, `g_max__uS` |
| $G_{\mathrm{lo}}, G_{\mathrm{hi}}$ | `prog_gamma` normalization bounds (independent of $G_{\min}, G_{\max}$) | uS | `prog_gamma.min_val`, `prog_gamma.max_val` |
| $\hat G$ | normalized conductance state | — | — |
| $V$ | device terminal voltage (runtime input) | V | `v__V` |
| $I_{\mathrm R}$ | device read current | uA | `RRAMDCOP.i__uA` |
| $\partial I_{\mathrm R}/\partial V$ | local differential conductance | uS | `RRAMDCOP.di_dv__uS` |
| $\alpha$ | I-V nonlinearity factor | 1/V | `nonlinearity_alpha` |
| $G_{\mathrm t}$ | target programming conductance (runtime input) | uS | `target_g__uS` |
| $d(t)$ | power-law drift gain | — | — |
| $\nu$ | drift exponent | — | `drift_decay_rate` |
| $t_0$ | reference drift time | s | `drift_t0` |
| $t$ | elapsed retention time (runtime input) | s | `t_elapsed` |
| $k, \theta$ | Gamma shape, scale | — | `prog_gamma` |
| $p_{\min}, p_{\max}$ | stuck-at-min / -max probabilities | — | `stuck_at` |
| $\mu_a, \sigma_a, p_{\mathrm{high}}$ | telegraph amplitude mean, std, active probability | uS, uS, — | `read_telegraph` |
| $\sigma_{\mathrm{th}}$ | thermal read-noise sigma | uS | `read_thermal__uS` |
| $C_{\mathrm{top}}, C_{\mathrm{bot}}$ | per-cell electrode capacitances | fF | `c_top__fF`, `c_bot__fF` |

## Assumptions, scope & validity

Stated assumptions of the current model:

- Conduction is quasi-static: a read returns the DC operating point at the applied voltage, with no within-read switching transient.
- The I-V law is symmetric in $V$ (the $\sinh$ form has odd symmetry); no rectifying / asymmetric conduction is modeled.
- $G_{\max}$ is a design ceiling enforced by clamping, representing external current limiting, not an intrinsic saturation of the device physics.
- Programming variation, drift, and stuck-at act at program time and are baked into the stored state; telegraph and thermal noise are resampled per read.
- The conductance-drift power law is a coarse placeholder: the drift exponent $\nu$ is a single state- and device-independent constant.
- The retention/drift time $t$ and its reference $t_0$ are kept in seconds, a separate quantity from the nanosecond compute-path time symbol of the shared [notation_conventions](../../../conventions/notation_conventions.md#electrical-and-physical-quantities).

TODO (domain author): give the quantitative validity boundaries — conductance and voltage ranges over which the $\sinh$ I-V holds, the temperature treatment of $G$ and $\alpha$ (currently temperature-independent in the read law), the retention-time range of the drift power law, and regimes where the model should not be trusted.

## Validation

TODO: link `validation/device` evidence — I-V and differential-conductance agreement, programming-write fixed points, and per-source noise-statistics checks against measured device data.

## References

TODO: cite the hyperbolic-sine RRAM I-V model, the power-law conductance-drift model, the state-dependent programming-variation model, and the random-telegraph-noise model.

---

- **Internals**: [rram internals](../../../internals/primitive/device/rram.md)
- **Validation**: TODO — `validation/device` (not yet written)
- **Configuration**: `api` (`RRAMConfig`, `RRAMPolicy`)
