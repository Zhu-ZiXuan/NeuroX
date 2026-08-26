# Triple-margin current SAR ADC

A triple-margin current-mode successive-approximation ADC, a member of the [single-ended current ADC family](family.md). A single current-mode sense amplifier (SA) is time-multiplexed over $b$ sequential comparisons — a binary search across the $2^{b_{\max}}-1$ nominal mid-point reference levels — producing a $b$-bit unsigned magnitude code from a single-ended magnitude current $I_{\mathrm{in}}$.

## Physical model

Each single comparison mirrors $I_{\mathrm{in}}$ and the step's reference $I_{\mathrm{ref}}$ into the sense amplifier, then a deterministic pre-gain $A = $ `margin_gain` amplifies the clean current difference $I_{\mathrm{in}} - I_{\mathrm{ref}}$ before the latch resolves its sign. The input-referred SA offset is a current-domain margin perturbation added **after** the pre-gain, so its effective value at the decision is divided by $A$ — the triple-margin benefit: a raw offset $\sigma$ acts as $\sigma / A$.

The mid-point thresholds are a runtime ladder $[\ldots,\ 2^{b_{\max}}-1]$ whose taps ascend along the last axis and whose leading dimensions broadcast right-aligned against $I_{\mathrm{in}}$. This topology compares against one wired threshold per binary-search node, so its reference count is $2^{b_{\max}}-1$ — a property of this circuit, not of the [family](family.md). The ladder is that full tap set at every resolution; the resolution $b$ lies in $[1,b_{\max}]$. Each step gathers one tap per element.

## Governing equations

The conversion runs a **truncated** binary search (MSB-first): the tree is always the $b_{\max}$-level one over the full ladder, and a $b$-bit conversion executes its first $b$ levels. At step $s$ (with $s = 0$ the MSB) the partial code resolved so far selects a mid-point reference $I_{\mathrm{ref},s}$ from it; the bit is the sign of the pre-gained clean margin plus the held offset,

$$D_s = \big[\,A\,(I_{\mathrm{in}} - I_{\mathrm{ref},s}) + \delta\,\big] > 0,$$

where $\delta$ is the static input-referred offset (comparator + coupling, zero when their policy toggles are off), held constant across all $b$ steps. The reference index tested at step $s$ is $\mathrm{prefix}\cdot 2^{\,b_{\max}-s} + 2^{\,b_{\max}-s-1} - 1$, where $\mathrm{prefix}$ is the high $s$ resolved bits — step 0 selects the ladder's central threshold $2^{\,b_{\max}-1}-1$ whatever $b$ is, and each later step bisects the surviving sub-interval. The executed levels are the leading bits of the $b_{\max}$-bit code, so the output is that code right-shifted,

$$\mathrm{code}_{b} = \big\lfloor \mathrm{code}_{b_{\max}} / 2^{\,b_{\max}-b} \big\rfloor \in [0,\ 2^{b}-1].$$

## Numerical method

The conversion performs $b$ sequential comparisons. Each step is a closed-form per-element reference selection and sign decision, with no inner iteration. Every step has the same decision latency, so the conversion latency is $b$ times that value.

## Energy model

Per sensing step the dynamic energy is one data-independent per-op constant plus the current-domain conduction drawn while the input and selected reference conduct across the rail,

$$E_{\mathrm{step},s} = E_{\mathrm{fixed}} + V_{\mathrm{rail}} \cdot (I_{\mathrm{in}} + I_{\mathrm{ref},s}) \cdot t_{\mathrm{cond}},$$

with $V_{\mathrm{rail}} = $ `v_rail__V` and $t_{\mathrm{cond}} = $ `t_conduct_per_step__ns` the common conduction window of every step ($1\,\mathrm{V} \cdot 1\,\mathrm{uA} \cdot 1\,\mathrm{ns} = 1\,\mathrm{fJ}$). A zero `t_conduct_per_step__ns` reduces the model to the pure fixed energy $b \cdot E_{\mathrm{fixed}}$ per element. Source-generation energy for the input and reference currents is outside this model.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
|---|---|---|---|
| comparator offset | static SA input-referred offset | static Gaussian current-domain margin, added after the pre-gain (effective $\sigma / A$), at fabricate | `comparator_offset_sigma__uA` |
| coupling mismatch | residual coupling-driven offset | static Gaussian current-domain margin, added after the pre-gain (effective $\sigma / A$), at fabricate | `coupling_mismatch_sigma__uA` |
| quantization | intrinsic binary-search resolution | deterministic threshold compare | `i_refs__uA` (per call) |

Both static offsets are sampled once at fabricate and held constant across the $b$ binary-search steps; each is zero when its policy toggle is off.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `bits` ($b$) | physical (maximum) magnitude resolution; a call requests any $b \in [1, b_{\max}]$ | — | $> 0$ | Design |
| `margin_gain` ($A$) | triple-margin pre-gain before the latch | — | $> 0$ | Design |
| `e_fixed_per_op__fJ` ($E_{\mathrm{fixed}}$) | data-independent per-step energy constant | fJ | $\geq 0$ | Design |
| `v_rail__V` ($V_{\mathrm{rail}}$) | rail the input and selected reference conduct across per step | V | $\geq 0$ | Design |
| `t_conduct_per_step__ns` ($t_{\mathrm{cond}}$) | conduction window shared by every step; zero ⇒ pure fixed energy | ns | $\geq 0$ | Design |
| `latency_per_step__ns` | decision latency of one search step | ns | $\geq 0$ | Design |
| `comparator_offset_sigma__uA` | static input-referred SA offset sigma | uA | $\geq 0$ | Measured |
| `coupling_mismatch_sigma__uA` | residual coupling-driven offset sigma | uA | $\geq 0$ | Measured |
| leakage / area | static PPA / spec fields | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $I_{\mathrm{in}}$ | single-ended magnitude input current | uA | `i_in__uA` |
| $I_{\mathrm{ref},s}$ | mid-point reference selected at step $s$ | uA | `i_refs__uA` (per call) |
| $A$ | triple-margin pre-gain | — | `margin_gain` |
| $V_{\mathrm{rail}}$ | per-step conduction rail | V | `v_rail__V` |
| $t_{\mathrm{cond}}$ | conduction window shared by every step | ns | `t_conduct_per_step__ns` |
| $\delta$ | static input-referred offset held across steps | uA | `comparator_offset__uA` + `coupling_offset__uA` |
| $D_s$ | decided bit at step $s$ (MSB-first) | — | code accumulation |
| $b$ | resolution (bits), per call | — | `bits` |

## Assumptions, scope & validity

Stated assumptions:

- The input is a single-ended non-negative magnitude.
- The single SA is time-shared across a set of columns; its fabricated `inst_shape` is the real shared sense-lane count, so functional codes are per-column while `inst_shape` sets only the PPA multiplicity.

TODO (domain author): the validity boundary of the triple-margin decision model and the input-range limits implied by the reference-level list.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the triple-margin current-mode sense-amplifier SAR topology.
