# RRAM cell

A two-terminal resistive-memory cell whose single state variable is the programmed conductance $G$. A read draws current from the stored $G$ at the applied terminal voltage $V$ under a symmetric hyperbolic-sine I-V law, yielding both the current and the local differential conductance; a write drives $G$ toward a target $G_{\mathrm t}$ subject to program-time and read-time non-idealities. Retention-time evolution is not modeled.

## Physical model

The device is abstracted as a programmable conductor with a single state variable, the per-cell conductance $G$, confined to a strictly positive working range $[G_{\min}, G_{\max}]$. $G_{\min}$ is an intrinsic floor of the device; $G_{\max}$ is a design ceiling set by external current limiting during programming, not an intrinsic device property. Two distinct operations act on the stored $G$:

- a **write** that drives the stored conductance toward a target value, subject to programming variation and stuck-at faults;
- a **read** that draws current from the present stored conductance under telegraph and thermal read noise.

The conduction itself is taken as instantaneous and quasi-static: a read returns the DC current at the applied terminal voltage with no transient switching dynamics within the read.

## Governing equations

**Read I-V law.** The conduction is a symmetric hyperbolic-sine law parameterized by the nonlinearity factor $\alpha$, reducing to an ohmic law in the linear limit $\alpha \to 0$:

$$I_{\mathrm R}(V) = \frac{G}{\alpha}\,\sinh(\alpha V), \qquad \frac{\partial I_{\mathrm R}}{\partial V} = G\,\cosh(\alpha V),$$

with the ohmic limit

$$I_{\mathrm R}(V) = G\,V, \qquad \frac{\partial I_{\mathrm R}}{\partial V} = G \quad (\alpha = 0).$$

**Programming write.** Programming composes a working-range clamp, state-dependent variation, stuck-at replacement, and a final clamp:

$$G \leftarrow \operatorname{clamp}\!\Big(\operatorname{stuck}\big(\Gamma(\operatorname{clamp}(G_{\mathrm t},\,G_{\min},\,G_{\max}))\big),\ G_{\min},\ G_{\max}\Big).$$

Here $\Gamma(\cdot)$ is the programming-variation map and $\operatorname{stuck}(\cdot)$ is the stuck-at map described below.

## Noise & non-idealities

The statistical forms below are the device's own non-ideality sources; the shared parameterization convention is in [nonideality](../nonideality.md).

- **Programming variation** (program time) — a multiplicative Gamma perturbation, normalized to unit mean, whose shape parameter $k$ depends on the normalized conductance state $\hat G = (G - G_{\mathrm{lo}})/(G_{\mathrm{hi}} - G_{\mathrm{lo}})$: $k(\hat G) = \max(k_{\mathrm{slope}}\,\hat G + k_{\mathrm{int}},\,0.1)$ at fixed scale $\theta$. The normalization bounds $G_{\mathrm{lo}}, G_{\mathrm{hi}}$ belong to the programming-variation model and are independent of the device working-range bounds $G_{\min}, G_{\max}$. The applied gain is $\gamma/\mathbb{E}[\gamma]$ with $\gamma \sim \operatorname{Gamma}(k,\theta)$, so the perturbation preserves the mean conductance and only injects state-dependent spread.
- **Stuck-at fault** (program time) — each cell is independently forced to $G_{\min}$ with probability $p_{\min}$ or to $G_{\max}$ with probability $p_{\max}$ (requiring $p_{\min}+p_{\max}<1$), else left unchanged.
- **Telegraph read noise** (read time) — random telegraph noise sampled on each read: a cell is in the active state with probability $p_{\mathrm{high}}$, and when active receives an additive perturbation of random sign and Gaussian-distributed amplitude (mean $\mu_a$, std $\sigma_a$).
- **Thermal read noise** (read time) — additive zero-mean Gaussian noise of std $\sigma_{\mathrm{th}}$ on the read conductance.

After the read-time sources, the read conductance is re-clamped to $[G_{\min}, G_{\max}]$.

## Parameters

| Parameter | Meaning | Unit | Constraint | [Source](../../../conventions/module_parameter.md) |
| --- | --- | --- | --- | --- |
| `g_min__uS` | minimum programmable conductance $G_{\min}$ | uS | $> 0$ | Measured |
| `g_max__uS` | maximum programmable conductance $G_{\max}$ (init kwarg, design ceiling) | uS | $> G_{\min}$ | Design |
| `nonlinearity_alpha` | hyperbolic-sine I-V factor $\alpha$ | 1/V | $\ge 0$ | Measured |
| `read_thermal__uS` | thermal read-noise sigma $\sigma_{\mathrm{th}}$ | uS | $\ge 0$ | Measured |
| `prog_gamma` ($k_{\mathrm{slope}}, k_{\mathrm{int}}, \theta$, norm range) | state-dependent programming-variation parameters | — | $\theta > 0$; $k_{\mathrm{int}} > 0$; $G_{\mathrm{hi}} > G_{\mathrm{lo}}$ | Measured |
| `read_telegraph` ($\mu_a, \sigma_a, p_{\mathrm{high}}$) | telegraph read-noise parameters | uS, uS, — | $\sigma_a \ge 0$; $0 \le p_{\mathrm{high}} \le 1$ | Measured |
| `stuck_at` ($p_{\min}, p_{\max}$) | stuck-at fault probabilities | — | $p_{\min}, p_{\max} \ge 0$; $p_{\min} + p_{\max} < 1$ | Measured |

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $G$ | per-cell programmed conductance | uS | `_g__uS`; read as `RramSnap.g__uS` |
| $G_{\min}, G_{\max}$ | working-range bounds | uS | `config.g_min__uS`, `_g_max__uS` |
| $G_{\mathrm{lo}}, G_{\mathrm{hi}}$ | `prog_gamma` normalization bounds (independent of $G_{\min}, G_{\max}$) | uS | `prog_gamma.min_val`, `prog_gamma.max_val` |
| $\hat G$ | normalized conductance state | — | — |
| $V$ | device terminal voltage (runtime input) | V | `v__V` |
| $I_{\mathrm R}$ | device read current | uA | `RramDcop.i__uA` |
| $\partial I_{\mathrm R}/\partial V$ | local differential conductance | uS | `RramDcop.di_dv__uS` |
| $\alpha$ | I-V nonlinearity factor | 1/V | `nonlinearity_alpha` |
| $G_{\mathrm t}$ | target programming conductance (runtime input) | uS | `target_g__uS` |
| $k, \theta$ | Gamma shape, scale | — | `prog_gamma` |
| $p_{\min}, p_{\max}$ | stuck-at-min / -max probabilities | — | `stuck_at` |
| $\mu_a, \sigma_a, p_{\mathrm{high}}$ | telegraph amplitude mean, std, active probability | uS, uS, — | `read_telegraph` |
| $\sigma_{\mathrm{th}}$ | thermal read-noise sigma | uS | `read_thermal__uS` |

## Assumptions, scope & validity

- Conduction is quasi-static: a read returns the DC operating point at the applied voltage, with no within-read switching transient.
- The I-V law is symmetric in $V$ (the $\sinh$ form has odd symmetry); no rectifying / asymmetric conduction is modeled.
- $G_{\max}$ is a design ceiling enforced by clamping, representing external current limiting, not an intrinsic saturation of the device physics.
- Programming variation and stuck-at act at program time and are baked into the stored state; telegraph and thermal noise are resampled per read.

Quantitative conductance, voltage, and temperature validity ranges are not established here; the read-law conductance and nonlinearity factor have no explicit temperature scaling.

## Validation

Evidence remains to be documented for I-V and differential-conductance agreement, programming-write fixed points, and per-source noise-statistics checks against measured device data.

## References

Citations for the hyperbolic-sine RRAM I-V model, the state-dependent programming-variation model, and the random-telegraph-noise model are not documented here.
