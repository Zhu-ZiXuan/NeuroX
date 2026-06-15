# MCS SAR ADC

## Summary / role

`McsSarAdc` is the $V_{\mathrm{cm}}$-based (Merged Capacitor Switching) differential SAR ADC: a successive-approximation converter with explicit cap-mismatch and comparator-noise modelling and calibrated multi-mode operation. It is the production SAR member of the [ADC family](README.md) and honours the family signed-code and floor contract in [base](base.md).

## Physical model

A differential MCS topology with bottom-plate sampling and two independent binary-weighted CDAC arrays, one per input leg. Physical CDAC depth is $b_{\max}$ bits; the active array has $b_{\max}-1$ binary-weighted caps plus a dummy unit cap. A conversion proceeds in three phases:

1. **Sample** - the bottom plates of the two CDAC arrays track the positive and negative inputs while the top plates float; on release the bottom plates snap to the common-mode voltage $V_{\mathrm{cm}}$, leaving each top plate at $V_{\mathrm{ref}} - V_{\mathrm{in}}$.
2. **MSB decision** - a free comparison of the two top plates resolves the most-significant bit; no MSB cap is needed because the differential topology resolves it directly. The comparison carries the static comparator offset and per-cycle thermal noise.
3. **SAR loop** - each subsequent cycle switches one cap on each leg from $V_{\mathrm{cm}}$ to $V_{\mathrm{ref}}$ or to ground according to the previous bit, moving each top plate by $\pm V_{\mathrm{cm}}\,C_k/C_{\mathrm{total}}$, until $b$ bits are resolved.

The modelled non-idealities are static per-cap Pelgrom mismatch (independent positive / negative legs), a static comparator offset, per-SAR-cycle comparator thermal noise, and kT/C sampling noise on the held top plates.

## Governing equations

Each SAR cycle perturbs the differential top-plate voltage by the merged-capacitor step

$$\Delta V_{\mathrm{top},k} = \pm\, V_{\mathrm{cm}}\,\frac{C_k}{C_{\mathrm{total}}},$$

where $C_k$ is the (mismatched, if the policy is on) capacitance of the cap switched on cycle $k$ and $C_{\mathrm{total}}$ the array total. The MSB is the sign of the free differential comparison; each subsequent bit is the sign of the running differential after the cycle's step, accumulated into the unsigned code. The unsigned code is then clamped to $[0,\ 2^{b}-1]$ and shifted by the zero code $2^{\,b-1}$ to the signed range $[-2^{\,b-1},\ 2^{\,b-1}-1]$ per the family contract in [base](base.md#signed-code-output-convention).

## Numerical method

The conversion is a successive-approximation loop of $b$ decision cycles after the free MSB comparison; each cycle is a closed-form step and a sign decision, with no inner iteration. The loop is exact for the modelled topology - it is not an iterative root find.

## Multi-mode operation

The operating point $(\mathrm{mode}, b)$ is per call:

- $\mathrm{mode}$ selects the reference-voltage entry $V_{\mathrm{ref}}$ from the strictly-decreasing `v_refs__V` tuple (entry 0 is the maximum, the calibration anchor).
- $b \le b_{\max}$ sets the active SAR depth; for $b < b_{\max}$ the loop engages only the top $b-1$ caps and the smaller caps stay idle.

Because every per-cell term scales with $\mathrm{mode}$ and $b$, one instance covers the full multi-mode envelope.

## Energy model

Per-conversion energy is the sum of:

- one-shot sampling energy charging the bottom plates from $V_{\mathrm{cm}}$ to $V_{\mathrm{in}}$;
- per-cycle MCS switching energy

$$E_k = \tfrac{1}{2}\,V_{\mathrm{ref}}^2\,C_k\left(1 - \frac{C_k}{C_{\mathrm{total}}}\right);$$

- reset / cap-mismatch dump energy at the end of conversion;
- a lump-sum overhead $E_{\mathrm{bootstrap}} + b\cdot E_{\mathrm{const}/\mathrm{bit}}$.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter | Policy switch |
|---|---|---|---|---|
| cap mismatch | per-cap area/oxide variation | static per-cap Pelgrom Gaussian on $C_k$ (independent legs), at fabricate | `cap_mismatch_sigma_relative` | `cap_mismatch` |
| comparator offset | static comparator input offset | static Gaussian threshold offset, at fabricate | `comparator_offset_sigma__V` | `comparator_offset` |
| comparator thermal noise | per-decision thermal noise | additive Gaussian per SAR cycle | `comparator_thermal_noise_sigma__V` | `comparator_thermal_noise` |
| sampling thermal noise | kT/C noise on held top plates | additive Gaussian at sample, sigma set by $T$ | (derived from $T$, $C$) | `sampling_thermal_noise` |
| quantization | intrinsic SAR resolution | deterministic (unbiased with training jitter) | $b$ | — |

The comparator thermal-noise sigma scales as $\sqrt{T}$ anchored at 300 K.

TODO (domain author): the kT/C sampling-noise sigma formula in terms of $k_B$, $T$, and $C_{\mathrm{total}}$, and citations for the MCS switching-energy and Pelgrom models.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `max_bits` | physical CDAC depth $b_{\max}$ | — | Design |
| `v_refs__V` | strictly-decreasing reference voltages (entry 0 = anchor) | V | Design |
| `clk_period__ns` | SAR clock period | ns | Design |
| `c_unit__fF` | unit-cap capacitance | fF | Design |
| `cap_mismatch_sigma_relative` | per-cap Pelgrom mismatch sigma | — | Measured |
| `comparator_offset_sigma__V` | static comparator-offset sigma | V | Measured |
| `comparator_thermal_noise_sigma__V` | per-cycle comparator-noise sigma (at 300 K) | V | Measured |
| `e_bootstrap__fJ` | per-conversion bootstrap energy | fJ | Design |
| `e_constant_per_bit__fJ` | per-bit constant energy overhead | fJ | Design |
| leakage / area | static PPA / spec fields | uW, um^2 | Design |

Per-op latency is not a parameter: it is derived as $(b+1)\cdot$ `clk_period__ns` from the runtime operating point. Provenance terms are defined in [parameter_provenance](../../parameter_provenance.md).

## Assumptions, scope & validity

Stated assumptions:

- The differential topology resolves the MSB by free comparison, so no dedicated MSB cap is modelled.
- The reference voltages are strictly decreasing, with entry 0 the calibration anchor.
- For $b < b_{\max}$ the smaller caps are idle and contribute no switching energy.

TODO (domain author): the validity range of the merged-capacitor step model (settling, parasitic coupling) and the operating envelope over which calibration is trusted.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the merged-capacitor-switching SAR topology and its energy model.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V^{+}, V^{-}$ | differential input legs | V | `v_pos__V`, `v_neg__V` |
| $V_{\mathrm{cm}}$ | common-mode voltage | V | derived |
| $V_{\mathrm{ref}}$ | selected reference voltage | V | `v_refs__V[mode]` |
| $V_{\mathrm{in}}$ | sampled input on a leg | V | sampled in `convert` |
| $C_k$ | capacitance of cap $k$ (mismatched after fabricate) | fF | per-leg cap arrays |
| $C_{\mathrm{total}}$ | total array capacitance | fF | derived |
| $b$ | runtime resolution (bits) | — | `adc_bits` |
| $b_{\max}$ | physical CDAC depth | — | `max_bits` |
| $E_k$ | per-cycle MCS switching energy | fJ | energy accounting |
| $E_{\mathrm{bootstrap}}, E_{\mathrm{const}/\mathrm{bit}}$ | energy overheads | fJ | `e_bootstrap__fJ`, `e_constant_per_bit__fJ` |
| $T$ | operating temperature | K | `T__K` |

---

- **Internals**: [mcs_sar internals](../../../internals/analog/adc/mcs_sar.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `McsSarAdcConfig`, `McsSarAdcPolicy` (see `api`)
- **Decisions**: N/A — no ADR governs this module.
