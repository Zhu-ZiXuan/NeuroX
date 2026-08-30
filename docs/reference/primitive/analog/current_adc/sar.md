# Current SAR ADC

A current-mode successive-approximation ADC, a member of the [single-ended current ADC family](family.md). A current comparator is time-multiplexed over $b$ sequential comparisons — a binary search across the $2^{b_{\max}}-1$ nominal mid-point reference levels — producing a $b$-bit unsigned magnitude code from a single-ended magnitude current $I_{\mathrm{in}}$.

## Physical model

Each step compares the positive input $I_{\mathrm{in}}$ against the negative reference input $I_{\mathrm{ref}}$. The static input-referred comparator offset $I_{\mathrm{os}}$ is defined as a threshold shift on the negative input: positive $I_{\mathrm{os}}$ raises the input current required to resolve a one. A concrete comparator that suppresses manufacturing offset supplies the correspondingly smaller equivalent $I_{\mathrm{os}}$; the SAR search itself does not model that internal circuit mechanism.

The mid-point thresholds are a runtime ladder $[\ldots,\ 2^{b_{\max}}-1]$ whose taps ascend along the last axis and whose leading dimensions broadcast right-aligned against $I_{\mathrm{in}}$. This topology compares against one wired threshold per binary-search node, so its reference count is $2^{b_{\max}}-1$ — a property of this circuit, not of the [family](family.md). The ladder is that full tap set at every resolution; the resolution $b$ lies in $[1,b_{\max}]$. Each step gathers one tap per element.

## Governing equations

The conversion runs a **truncated** binary search (MSB-first): the tree is always the $b_{\max}$-level one over the full ladder, and a $b$-bit conversion executes its first $b$ levels. At step $s$ (with $s = 0$ the MSB) the partial code resolved so far selects a mid-point reference $I_{\mathrm{ref},s}$ from it, then resolves

$$D_s = \big[\,I_{\mathrm{in}} \ge I_{\mathrm{ref},s} + I_{\mathrm{os}}\,\big].$$

The offset is sampled once per fabricated comparator instance and held across all $b$ steps; it is zero when comparator-offset sampling is disabled. The reference index tested at step $s$ is $\mathrm{prefix}\cdot 2^{\,b_{\max}-s} + 2^{\,b_{\max}-s-1} - 1$, where $\mathrm{prefix}$ is the high $s$ resolved bits — step 0 selects the ladder's central threshold $2^{\,b_{\max}-1}-1$ whatever $b$ is, and each later step bisects the surviving sub-interval. The executed levels are the leading bits of the $b_{\max}$-bit code, so the output is that code right-shifted,

$$\mathrm{code}_{b} = \big\lfloor \mathrm{code}_{b_{\max}} / 2^{\,b_{\max}-b} \big\rfloor \in [0,\ 2^{b}-1].$$

## Numerical method

The conversion performs $b$ sequential comparisons. Each step is a closed-form per-element reference selection and sign decision, with no inner iteration. Every step has the same decision latency, so the conversion latency is $b$ times that value.

## Noise & non-idealities

| Source | Physical origin | Statistical model | Parameter |
| --- | --- | --- | --- |
| comparator offset | static input-referred threshold offset | static Gaussian threshold shift on the negative reference input, at fabricate | `comparator_offset_sigma__uA` |
| quantization | intrinsic binary-search resolution | deterministic threshold compare | `i_refs__uA` (per call) |

The static offset is sampled once at fabricate and held constant across the $b$ binary-search steps.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
| --- | --- | --- | --- | --- |
| `bits` ($b_{\max}$) | physical magnitude output width | — | $> 0$ | Design |
| `latency_per_bit__ns` | decision latency of one output bit | ns | $\geq 0$ | Design |
| `comparator_offset_sigma__uA` | static input-referred comparator-offset sigma | uA | $\geq 0$ | Measured |
| leakage / area | static PPA / spec fields | uW, um^2 | $\geq 0$ | Design |

Provenance terms are defined in [module_parameter](../../../../conventions/module_parameter.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $I_{\mathrm{in}}$ | single-ended magnitude input current | uA | `i_in__uA` |
| $I_{\mathrm{ref},s}$ | mid-point reference selected at step $s$ | uA | `i_refs__uA` (per call) |
| $I_{\mathrm{os}}$ | static input-referred threshold offset held across steps | uA | `comparator_offset__uA` |
| $D_s$ | decided bit at step $s$ (MSB-first) | — | code accumulation |
| $b$ | active resolution, per call | — | `active_bits` |
| $b_{\max}$ | physical output bit width | — | `bits` |

## Assumptions, scope & validity

Stated assumptions:

- The input is a single-ended non-negative magnitude.
- The single SA is time-shared across a set of columns; its fabricated `inst_shape` is the real shared sense-lane count, so functional codes are per-column while `inst_shape` sets only the PPA multiplicity.

TODO (domain author): the input-range limits implied by the reference-level list.

## Validation

TODO - link validation evidence once written.

## References

TODO: cite the current-mode SAR topology.
