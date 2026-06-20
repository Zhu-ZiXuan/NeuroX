# RRAM Cell

## Summary

`RRAM` models one resistive-memory cell array: a two-terminal device whose programmed conductance is the analog weight a crossbar multiplies against. It sits at the bottom of the device→circuit→architecture stack as a storage and current-source primitive, conducting between its top electrode (BL side) and its bottom electrode (internal-node side) and exposing its read I-V and the local differential conductance to a consuming circuit. This document specifies the conductance value domain, the current-voltage law, the programming write model, and the programming-time and read-time non-ideality stack. The cell carries no array geometry, encoding, or readout — those belong to the consuming circuit.

## Physical model

The device is abstracted as a programmable conductor with a single state variable, the per-cell conductance $G$, confined to a working range $[G_{\min}, G_{\max}]$. $G_{\min}$ is an intrinsic floor of the device; $G_{\max}$ is a design ceiling set by external current limiting during programming, not an intrinsic device property. Two physically distinct surfaces act on $G$:

- a **programming** surface (a write) that drives the stored conductance toward a target value, subject to programming variation, conductance drift over the elapsed retention time, and stuck-at faults;
- a **read** surface that, at VMM time, draws current from the present stored conductance under telegraph and thermal read noise.

The conduction itself is taken as instantaneous and quasi-static: a read returns the DC current at the applied terminal voltage with no transient switching dynamics within the read. The per-cell electrode parasitic capacitances ($C_{\mathrm{top}}$ on the BL side, $C_{\mathrm{bot}}$ on the internal-node side) are exported for the consuming circuit's energy model and do not enter the conduction law.

## Governing equations

**Read I-V law.** The conduction is a symmetric hyperbolic-sine law parameterized by the nonlinearity factor $\alpha$, reducing to an ohmic law in the linear limit $\alpha \to 0$:

$$I_{\mathrm R}(V) = \frac{G}{\alpha}\,\sinh(\alpha V), \qquad \frac{\partial I_{\mathrm R}}{\partial V} = G\,\cosh(\alpha V),$$

with the ohmic limit

$$I_{\mathrm R}(V) = G\,V, \qquad \frac{\partial I_{\mathrm R}}{\partial V} = G \quad (\alpha = 0).$$

A device evaluation returns both the current and the local differential conductance $\partial I_{\mathrm R}/\partial V$, the latter consumed as the cell's contribution to a circuit-level Jacobian.

**Programming write.** A write to a target conductance $G_{\mathrm t}$ after elapsed time $t$ is the composition, in order, of a working-range clamp, state-dependent programming variation, a power-law drift gain, a stuck-at replacement, and a final clamp:

$$G \leftarrow \operatorname{clamp}\!\Big(\operatorname{stuck}\big(d(t)\cdot\Gamma(\operatorname{clamp}(G_{\mathrm t},\,G_{\min},\,G_{\max}))\big),\ G_{\min},\ G_{\max}\Big),$$

where $\Gamma(\cdot)$ is the state-dependent programming-variation map (§Noise), $\operatorname{stuck}(\cdot)$ the stuck-at map, and the drift gain is a power law applied only once the elapsed time exceeds the reference time $t_0$:

$$d(t) = \left(\frac{t}{t_0}\right)^{-\nu} \quad (t > t_0,\ \nu > 0), \qquad d(t) = 1 \quad \text{otherwise}.$$

## Numerical method

N/A — the read I-V and its derivative are evaluated in closed form; the cell holds no solver of its own. The consuming circuit assembles these per-cell currents and differential conductances into its operating-point solve.

## Noise & non-idealities

Each source is independently switched by a per-run policy flag; with the flag off the source is the identity map. The statistical forms below are the device-level kernels; the shared parameterization convention is in [notation_conventions](../notation_conventions.md#noise-model-conventions).

- **Programming variation** (`prog_gamma`, program time) — a multiplicative Gamma perturbation, normalized to unit mean, whose shape parameter $k$ depends on the normalized conductance state $\hat G = (G - G_{\mathrm{lo}})/(G_{\mathrm{hi}} - G_{\mathrm{lo}})$: $k(\hat G) = \max(k_{\mathrm{slope}}\,\hat G + k_{\mathrm{int}},\,0.1)$ at fixed scale $\theta$. The normalization bounds $G_{\mathrm{lo}}, G_{\mathrm{hi}}$ are the `prog_gamma` config's own `min_val`/`max_val` and are independent of the device working-range bounds $G_{\min}, G_{\max}$ (they coincide only when a preset sets them equal). The applied gain is $\gamma/\mathbb{E}[\gamma]$ with $\gamma \sim \operatorname{Gamma}(k,\theta)$, so the perturbation preserves the mean conductance and only injects state-dependent spread.
- **Stuck-at fault** (`stuck_at`, program time) — each cell is independently forced to $G_{\min}$ with probability $p_{\min}$ or to $G_{\max}$ with probability $p_{\max}$ (requiring $p_{\min}+p_{\max}<1$), else left unchanged.
- **Conductance drift** (program time) — the power-law gain $d(t)$ above; always applied when $\nu>0$ and $t>t_0$ (governed by the device parameters, not a policy flag).
- **Telegraph read noise** (`read_telegraph`, read time) — random telegraph noise added per read snapshot: a cell is in the active state with probability $p_{\mathrm{high}}$, and when active receives an additive perturbation of random sign and Gaussian-distributed amplitude (mean $\mu_a$, std $\sigma_a$).
- **Thermal read noise** (`read_thermal`, read time) — additive zero-mean Gaussian noise of std $\sigma_{\mathrm{th}}$ on the read conductance.

After the read-time sources, the snap conductance is re-clamped to $[G_{\min}, G_{\max}]$.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `g_min__uS` | minimum programmable conductance $G_{\min}$ | uS | Measured |
| `g_max__uS` | maximum programmable conductance $G_{\max}$ (init kwarg, design ceiling) | uS | Design |
| `nonlinearity_alpha` | hyperbolic-sine I-V factor $\alpha$ | 1/V | Measured |
| `drift_decay_rate` | power-law drift exponent $\nu$ | — | Measured |
| `drift_t0` | reference drift time $t_0$ | s | Measured |
| `c_top__fF` | top-electrode (BL-side) parasitic capacitance per cell $C_{\mathrm{top}}$ | fF | Process |
| `c_bot__fF` | bottom-electrode (internal-node-side) parasitic capacitance per cell $C_{\mathrm{bot}}$ | fF | Process |
| `read_thermal__uS` | thermal read-noise sigma $\sigma_{\mathrm{th}}$ | uS | Measured |
| `prog_gamma` ($k_{\mathrm{slope}}, k_{\mathrm{int}}, \theta$, norm range) | state-dependent programming-variation parameters | — | Measured |
| `read_telegraph` ($\mu_a, \sigma_a, p_{\mathrm{high}}$) | telegraph read-noise parameters | uS, uS, — | Measured |
| `stuck_at` ($p_{\min}, p_{\max}$) | stuck-at fault probabilities | — | Measured |

Provenance terms are defined in [parameter_provenance](../parameter_provenance.md). How to obtain values for a new chip: `guides/calibration`; file-level schema: `api`.

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
- The conductance-drift power law is a placeholder, not yet detailed-modelled: the drift exponent $\nu$ is a single state- and device-independent constant, and its state and device dependence is deferred.
- The retention/drift time $t$ and its reference $t_0$ are kept in seconds, a separate quantity from the nanosecond compute-path time symbol of the shared [notation_conventions](../notation_conventions.md#electrical-and-physical-quantities).

TODO (domain author): give the quantitative validity boundaries — conductance and voltage ranges over which the $\sinh$ I-V holds, the temperature treatment of $G$ and $\alpha$ (currently temperature-independent in the read law), the retention-time range of the drift power law, and regimes where the model should not be trusted.

## Validation

TODO: link `validation/device` evidence — I-V and differential-conductance agreement, programming-write fixed points, and per-source noise-statistics checks against measured device data.

## References

TODO: cite the hyperbolic-sine RRAM I-V model, the power-law conductance-drift model, the state-dependent programming-variation model, and the random-telegraph-noise model.

---

- **Internals**: [rram internals](../../internals/device/rram.md)
- **Validation**: TODO — `validation/device` (not yet written)
- **Configuration**: `api` (`RRAMConfig`, `RRAMPolicy`)
- **Decisions**: N/A — no ADR governs this device.
