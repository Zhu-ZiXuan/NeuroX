# Switch-cap bank

## Summary / role

The `SwitchCap` is a bottom-plate-sampled capacitor bank that performs passive charge-share averaging across a fixed set of weighted unit caps. It is a structural building block a readout chain composes directly — one concrete block, not a polymorphic family. The bank itself is encoding-agnostic, so the semantic meaning of its weights is set by the composing circuit, not by the bank.

## Physical model

A bank of `n_caps` capacitors shares one common plate. Cap $k$ has nominal capacitance $C_k = C_{\mathrm{unit}}\,a_k$, where $a_k$ is a positive per-cap weight and $C_{\mathrm{unit}}$ the unit-cell capacitance. On a sample each cap is charged to its input voltage on its bottom plate; on release the common plate settles to the charge-weighted average of the sampled voltages. The two physical non-idealities are static per-unit-cell capacitance mismatch (Pelgrom) and per-cap kT/C settling noise sampled onto each held input ($\sigma_k = \sqrt{k_B T / C_k}$).

## Governing equations

The bank samples a per-cap voltage vector $\{V_k\}$, perturbs each held voltage by an independent kT/C settling-noise sample before charge-sharing, and returns the passive charge-share average

$$V_{\mathrm{out}} = \frac{\sum_k C_k \left(V_k + n_{kT/C,k}\right)}{\sum_k C_k}, \qquad n_{kT/C,k} \sim \mathcal{N}\!\left(0,\ \sigma_k^2\right),\quad \sigma_k = \sqrt{k_B T / C_k},$$

with $n_{kT/C,k}$ the per-cap kT/C settling-noise sample on the held input, and each $C_k$ the fabricated per-unit-cell (Pelgrom) mismatched capacitance about its nominal $C_{\mathrm{unit}}\,a_k$.

## Numerical method

N/A - the charge-share average is closed-form per call; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| cap mismatch | per-unit-cell area/oxide variation | static per-unit-cell Pelgrom-scaled Gaussian on $C_k$, sampled at fabricate | `cap_mismatch_sigma_relative` |
| sampling thermal noise | per-cap kT/C settling noise | additive zero-mean Gaussian on each held input $V_k$, sigma $\sigma_k = \sqrt{k_B T / C_k}$ per cap | (derived from $T$, $C_k$) |

Cap mismatch is static — a fixed offset frozen at fabrication; kT/C noise is dynamic — an independent fresh draw per cap on each sampling event. Pelgrom area scaling sets the mismatch magnitude: the relative capacitance-matching sigma falls as the inverse square root of capacitor area, so a cap of weight $a_k$ (occupying $a_k$ unit cells) has relative mismatch sigma $\sigma_u / \sqrt{a_k}$ and absolute sigma $\sigma_u\,C_{\mathrm{unit}}\sqrt{a_k}$, where $\sigma_u$ is the per-unit-cell relative sigma `cap_mismatch_sigma_relative`. The kT/C sigma $\sigma_k = \sqrt{k_B T / C_k}$ rises with temperature $T$ and falls with per-cap capacitance $C_k$.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `c_unit__fF` | unit-cell capacitance $C_{\mathrm{unit}}$ | fF | Design |
| `cap_mismatch_sigma_relative` | per-unit-cell relative mismatch sigma $\sigma_u$ (Pelgrom) | — | Measured |
| cap weights $a_k$ | per-cap positional weights (deployment-bound, length `n_caps`) | — | Design |
| `energy_per_sample_overhead__fJ` | per-sample energy overhead | fJ | Design |
| leakage / area / latency | static PPA / spec fields | uW, um^2, ns | Design |

Provenance terms are defined in [module_parameter](../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $C_k$ | fabricated capacitance of cap $k$ | fF | `c__fF` |
| $C_{\mathrm{unit}}$ | unit-cell capacitance | fF | `c_unit__fF` |
| $a_k$ | per-cap positional weight | — | `cap_weights` |
| $V_k$ | sampled voltage on cap $k$ | V | `v_in__V` |
| $V_{\mathrm{out}}$ | charge-share output | V | `v_out__V` |
| $n_{kT/C,k}$ | per-cap kT/C settling-noise sample added to $V_k$ | V | — |
| $\sigma_k$ | per-cap kT/C noise sigma, $\sqrt{k_B T / C_k}$ | V | — |
| $\sigma_u$ | per-unit-cell relative mismatch sigma | — | `cap_mismatch_sigma_relative` |
| $T$ | operating temperature | K | `T__K` |
| $k_B$ | Boltzmann constant | J/K | `K_BOLTZMANN__J_per_K` |

## Assumptions, scope & validity

Stated assumptions:

- The averaging is an ideal passive charge share over a fixed set of positive-weighted caps.
- The bank performs no value arithmetic beyond the charge-weighted average; the encoding meaning of the weights lives in the composing circuit.

TODO (domain author): the validity boundary of the ideal-charge-share assumption (parasitic top-plate capacitance, incomplete settling, charge injection) and the regime where it breaks down.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the bottom-plate-sampling charge-share and the Pelgrom mismatch model.

---

- **Internals**: [switch_cap internals](../../internals/analog/switch_cap.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `SwitchCapConfig`, `SwitchCapPolicy` (see `api`)
