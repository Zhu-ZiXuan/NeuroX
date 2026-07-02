# Switch-cap bank

## Summary / role

The `SwitchCap` is a bottom-plate-sampled capacitor bank that performs passive charge-share averaging across a fixed set of weighted unit caps. In the readout chain it realises the per-digit positional accumulation (signal leg) and the single-cap baseline (reference leg); the bank itself is encoding-agnostic, so the semantic meaning of its weights is set by the composing circuit, not by the bank. It is a single concrete leaf block, not a polymorphic family.

## Physical model

A bank of $n_{\mathrm{caps}}$ capacitors shares one common plate. Cap $k$ has nominal capacitance $C_k = C_{\mathrm{unit}}\,a_k$, where $a_k$ is a positive per-cap weight and $C_{\mathrm{unit}}$ the unit-cell capacitance. On a sample each cap is charged to its input voltage on its bottom plate; on release the common plate settles to the charge-weighted average of the sampled voltages. The two physical non-idealities are static per-unit-cell capacitance mismatch (Pelgrom) and per-cap kT/C settling noise sampled onto each held input ($\sigma_k = \sqrt{k_B T / C_k}$).

## Governing equations

The bank samples a per-cap voltage vector $\{V_k\}$, perturbs each held voltage by an independent kT/C settling-noise sample before charge-sharing, and returns the passive charge-share average

$$V_{\mathrm{out}} = \frac{\sum_k C_k \left(V_k + n_{kT/C,k}\right)}{\sum_k C_k}, \qquad n_{kT/C,k} \sim \mathcal{N}\!\left(0,\ \sigma_k^2\right),\quad \sigma_k = \sqrt{k_B T / C_k},$$

with $C_k = C_{\mathrm{unit}}\,a_k$ and $n_{kT/C,k}$ the optional per-cap settling-noise sample on the held input. With the static mismatch policy on, each $C_k$ is the fabricated mismatched value rather than its nominal; otherwise $C_k$ is nominal.

## Numerical method

N/A - the charge-share average is closed-form per call; no iteration.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter | Policy switch |
|---|---|---|---|---|
| cap mismatch | per-unit-cell area/oxide variation | static per-unit-cell Pelgrom-scaled Gaussian on $C_k$, sampled at fabricate | `cap_mismatch_sigma_relative` | `cap_mismatch` |
| sampling thermal noise | per-cap kT/C settling noise | additive zero-mean Gaussian on each held input $V_k$, sigma $\sigma_k = \sqrt{k_B T / C_k}$ per cap | (derived from $T$, $C_k$) | `sampling_thermal_noise` |

Cap mismatch is static (sampled once per fabricate, per instance); kT/C noise is dynamic (sampled per call, per cap). The kT/C sigma depends on the operating temperature $T$ and the per-cap capacitance $C_k$; $T$ is bound at construction rather than as a design parameter.

TODO (domain author): write the Pelgrom scaling law relating `cap_mismatch_sigma_relative` to unit-cell area, and the kT/C sigma formula in terms of $k_B$, $T$, and per-cap capacitance $C_k$, with citations.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `c_unit__fF` | unit-cell capacitance $C_{\mathrm{unit}}$ | fF | Design |
| `cap_mismatch_sigma_relative` | per-unit-cell Pelgrom mismatch sigma | — | Measured |
| cap weights $a_k$ | per-cap positional weights (deployment-bound, length $n_{\mathrm{caps}}$) | — | Design |
| `energy_per_sample_overhead__fJ` | per-sample energy overhead | fJ | Design |
| leakage / area / latency | static PPA / spec fields | uW, um^2, ns | Design |

Provenance terms are defined in [module_parameter](../../conventions/module_parameter.md). The cap weights are a structural deployment property of the bank (the digit-encoding it is committed to), distinct from the per-cell physical knobs.

## Assumptions, scope & validity

Stated assumptions:

- The averaging is an ideal passive charge share over a fixed set of positive-weighted caps; the weight template is fixed across re-fabricates.
- The bank performs no value arithmetic beyond the charge-weighted average; the encoding meaning of the weights lives in the composing circuit.

TODO (domain author): the validity boundary of the ideal-charge-share assumption (parasitic top-plate capacitance, incomplete settling, charge injection) and the regime where it breaks down.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the bottom-plate-sampling charge-share and the Pelgrom mismatch model.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $C_k$ | capacitance of cap $k$ | fF | `nominal_c__fF` (mismatched after fabricate) |
| $C_{\mathrm{unit}}$ | unit-cell capacitance | fF | `c_unit__fF` |
| $a_k$ | per-cap positional weight | — | `cap_weights` |
| $V_k$ | sampled voltage on cap $k$ | V | `sample_and_accumulate` input |
| $V_{\mathrm{out}}$ | charge-share output | V | `sample_and_accumulate` output |
| $n_{kT/C,k}$ | per-cap per-call kT/C noise sample on $V_k$ | V | sampled at sample time |
| $T$ | operating temperature | K | `T__K` |

---

- **Internals**: [switch_cap internals](../../internals/analog/switch_cap.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `SwitchCapConfig` (see `api`)
- **Decisions**: N/A — no ADR governs this module.
