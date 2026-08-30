# Switch-cap bank

## Physical model

A bank of $N$ capacitors shares one common plate. Cap $k$ has nominal capacitance $C_k = C_{\mathrm{unit}}\,a_k$, where $a_k$ is a positive per-cap weight and $C_{\mathrm{unit}}$ the unit-cell capacitance. On a sample each cap is charged to its input voltage on its bottom plate; on release the common plate settles to the charge-weighted average of the sampled voltages. The two modelled non-idealities are static per-unit-cell capacitance mismatch (Pelgrom) and per-cap kT/C settling noise on each held input.

## Governing equations

The bank samples a per-cap voltage vector $\{V_k\}$, perturbs each held voltage by an independent kT/C settling-noise sample before charge-sharing, and returns the passive charge-share average

$$V_{\mathrm{out}} = \frac{\sum_k C_k \left(V_k + n_{kT/C,k}\right)}{\sum_k C_k}, \qquad n_{kT/C,k} \sim \mathcal{N}\!\left(0,\ \sigma_k^2\right),\quad \sigma_k = \sqrt{k_B T / C_k},$$

with $n_{kT/C,k}$ the per-cap kT/C settling-noise sample on the held input, and each $C_k$ the fabricated per-unit-cell (Pelgrom) mismatched capacitance about its nominal $C_{\mathrm{unit}}\,a_k$.

## Numerical method

N/A — the charge-share average is closed-form; no iteration.

## Energy model

Sampling charges each cap to its per-cap voltage: charging cap $k$ from ground to $V_k$ deposits $\tfrac{1}{2} C_k V_k^2$. Summed over the bank, this sampled-charge energy is the dominant, signal-dependent term,

$$E_{\mathrm{caps}} = \tfrac{1}{2}\sum_k C_k V_k^2,$$

evaluated at the fabricated capacitances $C_k$ and the sampled voltages $V_k$ (before the kT/C perturbation). A fixed per-sample switching overhead $E_{\mathrm{overhead}}$, independent of the sampled voltages, adds to it for the per-sample dynamic energy

$$E = E_{\mathrm{caps}} + E_{\mathrm{overhead}}.$$

In the consistent unit set $\mathrm{fF}\times\mathrm{V}^2 = \mathrm{fJ}$.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
| --- | --- | --- | --- |
| cap mismatch | per-unit-cell area/oxide variation | static per-unit-cell Pelgrom-scaled Gaussian on $C_k$, sampled at fabricate | $\sigma_u$ |
| sampling thermal noise | per-cap kT/C settling noise | additive zero-mean Gaussian on each held input $V_k$, sigma $\sigma_k = \sqrt{k_B T / C_k}$ per cap | $\sigma_k$ (derived from $T$, $C_k$) |

Cap mismatch is static — a fixed offset frozen at fabrication; kT/C noise is dynamic — an independent fresh draw per cap on each sampling event. Pelgrom area scaling sets the mismatch magnitude: the relative capacitance-matching sigma falls as the inverse square root of capacitor area, so a cap of weight $a_k$ (occupying $a_k$ unit cells) has relative mismatch sigma $\sigma_u / \sqrt{a_k}$ and absolute sigma $\sigma_u\,C_{\mathrm{unit}}\sqrt{a_k}$, where $\sigma_u$ is the per-unit-cell relative sigma.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
| --- | --- | --- | --- | --- |
| `c_unit__fF` ($C_{\mathrm{unit}}$) | unit-cell capacitance | fF | $> 0$ | Design |
| `cap_mismatch_sigma_relative` ($\sigma_u$) | per-unit-cell relative mismatch sigma (Pelgrom) | — | $\geq 0$ | Measured |
| `cap_weights` ($a_k$) | per-cap weight on $C_{\mathrm{unit}}$, length $N$ | — | $> 0$ | Design |
| `energy_per_sample_overhead__fJ` ($E_{\mathrm{overhead}}$) | per-sample energy overhead, independent of the sampled voltages | fJ | $\geq 0$ | Design |
| leakage / area | static PPA / spec fields | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $N$ | number of caps in the bank | — | `_cap_num` |
| $C_k$ | fabricated capacitance of cap $k$ | fF | `_c__fF` |
| $C_{\mathrm{unit}}$ | unit-cell capacitance | fF | `c_unit__fF` |
| $a_k$ | per-cap weight on $C_{\mathrm{unit}}$ | — | `cap_weights` |
| $V_k$ | sampled voltage on cap $k$ | V | `v_in__V` |
| $V_{\mathrm{out}}$ | charge-share output | V | `v_out__V` |
| $n_{kT/C,k}$ | per-cap kT/C settling-noise sample added to $V_k$ | V | — |
| $\sigma_k$ | per-cap kT/C noise sigma, $\sqrt{k_B T / C_k}$ | V | `sigma__V` |
| $\sigma_u$ | per-unit-cell relative mismatch sigma | — | `cap_mismatch_sigma_relative` |
| $T$ | operating temperature | K | `T__K` |
| $k_B$ | Boltzmann constant | J/K | `K_BOLTZMANN__J_per_K` |
| $E_{\mathrm{caps}}$ | sampled-charge energy per sample, $\tfrac{1}{2}\sum_k C_k V_k^2$ | fJ | `e_caps__fJ` |
| $E_{\mathrm{overhead}}$ | per-sample switching-energy overhead | fJ | `energy_per_sample_overhead__fJ` |
| $E$ | per-sample dynamic energy, $E_{\mathrm{caps}} + E_{\mathrm{overhead}}$ | fJ | `dynamic_energy__fJ` |

## Assumptions, scope & validity

Stated assumptions:

- The averaging is an ideal passive charge share over a fixed set of positive-weighted caps.
- The bank performs no value arithmetic beyond the charge-weighted average; the weights carry no encoding semantics in the model.

TODO (domain author): the validity boundary of the ideal-charge-share assumption (parasitic top-plate capacitance, incomplete settling, charge injection) and the regime where it breaks down.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the bottom-plate-sampling charge-share and the Pelgrom mismatch model.
