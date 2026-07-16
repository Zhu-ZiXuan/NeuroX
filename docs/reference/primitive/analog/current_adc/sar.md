# Triple-margin current SAR ADC

A triple-margin current-mode successive-approximation ADC, a member of the [current ADC family](family.md). A single current-mode sense amplifier (SA) is time-multiplexed over $b$ sequential comparisons — a binary search across $2^{b}-1$ nominal mid-point reference levels — producing a $b$-bit unsigned magnitude code from a single-ended magnitude current $I_{\mathrm{in}}$.

## Physical model

Each single comparison mirrors $I_{\mathrm{in}}$ and the step's reference $I_{\mathrm{ref}}$ through input mirrors sized $n = $ `input_mirror_ratio` times the reference legs, then a deterministic pre-gain $A = $ `margin_gain` amplifies the clean current difference $I_{\mathrm{in}} - I_{\mathrm{ref}}$ before the latch resolves its sign. The input-referred SA offset is a current-domain margin perturbation added **after** the pre-gain, so its effective value at the decision is divided by $A$ — the triple-margin benefit: a raw offset $\sigma$ acts as $\sigma / A$.

The $2^{b}-1$ nominal mid-point thresholds `ref_levels__uA` are a config tuple; the ADC reads them directly and self-holds no external reference.

## Governing equations

The conversion runs a $b$-step binary search (MSB-first). At step $s$ (with $s = 0$ the MSB) the partial code resolved so far selects a mid-point reference $I_{\mathrm{ref},s}$; the bit is the sign of the pre-gained clean margin plus the held offset,

$$D_s = \big[\,A\,(I_{\mathrm{in}} - I_{\mathrm{ref},s}) + \delta\,\big] > 0,$$

where $\delta$ is the static input-referred offset (comparator + coupling, zero when their policy toggles are off), held constant across all $b$ steps. The reference index tested at step $s$ is $\mathrm{prefix}\cdot 2^{\,b-s} + 2^{\,b-s-1} - 1$, where $\mathrm{prefix}$ is the high $s$ resolved bits — step 0 selects the central threshold and each later step bisects the surviving sub-interval. The output is the accumulated unsigned code in $[0,\ 2^{b}-1]$.

## Numerical method

The conversion performs $b$ comparisons in a method-internal Python loop; each step is a closed-form reference select and a sign decision, with no inner iteration. The sub-comparisons are not separate profiled leaves, so the whole conversion emits exactly one dynamic-energy event and one latency event.

## Energy model

Per sensing step the dynamic energy is the data-dependent regeneration the SA's sized mirror controls, plus one data-independent per-op constant,

$$E_{\mathrm{step}} = V_{\mathrm{rail,SA}} \cdot n \cdot (I_{\mathrm{in}}^{+} + I_{\mathrm{ref}}^{+}) \cdot t_{\mathrm{eff}} + E_{\mathrm{fixed}},$$

with $n = $ `input_mirror_ratio` the regeneration legs, $I^{+}$ the non-negative-clamped currents, $V_{\mathrm{rail,SA}}$ the SA-leg overdrive, and $t_{\mathrm{eff}}$ the effective conduction time. Control-based attribution bills only the regeneration path the ADC's sized mirror controls; the unity input and reference legs are owned and billed by the upstream blocks that source them, so neither is re-billed here. The step energies are summed into one per-conversion event. Conversion latency is $\sum_s$ `step_latency__ns`.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| comparator offset | static SA input-referred offset | static Gaussian current-domain margin, added after the pre-gain (effective $\sigma / A$), at fabricate | `comparator_offset_sigma__uA` |
| coupling mismatch | residual coupling-driven offset | static Gaussian current-domain margin, added after the pre-gain (effective $\sigma / A$), at fabricate | `coupling_mismatch_sigma__uA` |
| quantization | intrinsic binary-search resolution | deterministic threshold compare | `ref_levels__uA` |

Both static offsets are sampled once at fabricate and held constant across the $b$ binary-search steps; each is zero when its policy toggle is off. The mirror-ratio mismatch (`mirror_mismatch_sigma_relative`) and reference-level tracking (`replica_threshold_variation`) are wired in the policy but not yet modelled.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `n_bits` ($b$) | output magnitude resolution | — | $> 0$ | Design |
| `margin_gain` ($A$) | triple-margin pre-gain before the latch | — | $> 0$ | Design |
| `input_mirror_ratio` ($n$) | regeneration mirror ratio vs the unity legs | — | $> 0$ | Design |
| `ref_levels__uA` ($I_{\mathrm{ref},c}$) | $2^{b}-1$ nominal mid-point thresholds | uA | strictly increasing | Calibrated (physical data) |
| `v_rail_sa__V` | SA-leg overdrive the regeneration current is pulled across | V | $\geq 0$ | Design |
| `t_eff__ns` | effective conduction time the regeneration is drawn over | ns | $\geq 0$ | Design |
| `e_fixed_per_op__fJ` | data-independent per-op energy constant | fJ | $\geq 0$ | Design |
| `step_latency__ns` | per-step decision latency, one entry per step | ns | $\geq 0$ | Design |
| `comparator_offset_sigma__uA` | static input-referred SA offset sigma | uA | $\geq 0$ | Measured |
| `coupling_mismatch_sigma__uA` | residual coupling-driven offset sigma | uA | $\geq 0$ | Measured |
| `mirror_mismatch_sigma_relative` | relative sigma on the mirror ratios | — | $\geq 0$ | Measured |
| leakage / area | static PPA / spec fields | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $I_{\mathrm{in}}$ | single-ended magnitude input current | uA | `i_in__uA` |
| $I_{\mathrm{ref},s}$ | mid-point reference selected at step $s$ | uA | `ref_levels__uA` |
| $A$ | triple-margin pre-gain | — | `margin_gain` |
| $n$ | regeneration mirror ratio | — | `input_mirror_ratio` |
| $\delta$ | static input-referred offset held across steps | uA | `comparator_offset__uA` + `coupling_offset__uA` |
| $D_s$ | decided bit at step $s$ (MSB-first) | — | code accumulation |
| $b$ | resolution (bits) | — | `n_bits` / `adc_bits` |

## Assumptions, scope & validity

Stated assumptions:

- The input is a single-ended non-negative magnitude; the sign is reattached by the caller.
- The single SA is time-shared across a set of columns; its fabricated `inst_shape` is the real shared sense-lane count, so functional codes are per-column while `inst_shape` sets only the PPA multiplicity and the serial-latency time-multiplex factor.

TODO (domain author): the validity boundary of the triple-margin decision model and the input-range limits implied by the reference-level list.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the triple-margin current-mode sense-amplifier SAR topology.

---

- **Internals**: [sar internals](../../../../internals/primitive/analog/current_adc/sar.md)
- **Validation**: TODO - validation evidence not yet written
- **Configuration**: `SarCurrentAdcConfig`, `SarCurrentAdcPolicy` (see `api`)
