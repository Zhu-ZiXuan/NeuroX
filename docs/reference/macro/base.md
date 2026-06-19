# Macro Abstract Contract

## Summary

A macro is the architectural unit that turns one or more [crossbar tiles](../xbar/README.md) into a complete quantised-integer matrix multiply. It models the simulated hardware one level above the tile: the tile carries a single primitive analog VMM over its native integer value domain, and the macro is what decomposes a high-precision logical weight and activation onto that domain, schedules the per-tile reads, and recombines the partial results into one integer dot product. This document specifies the macro contract that every family member satisfies; the xbar-tile family that realizes it is in [xbar/](xbar/README.md).

## Physical model

The macro models no silicon of its own. It owns no cells, no interconnect, no readout — those belong to the tiles and the digital reduction blocks it composes. What the macro adds is the *architecture*: the rule by which a logical weight matrix that exceeds one tile's value range or row/column count is spread across tiles, and the dual rule by which the per-tile results are summed back. Physically this corresponds to a chip region holding an array of crossbar tiles plus the digital adders that fold their outputs; the macro is the abstraction of that region's organization.

The macro presents a single integer matrix multiply to its consumer. Internally the multiply is realized as the sum of per-tile primitive VMMs over a decomposed value domain, but the contract is value-domain only: it states the integer ranges accepted and the integer (pre-requantize) result produced, not the physics of any constituent read.

## Governing equations

The macro computes an integer dot product matching exact matrix multiplication over its accepted value domain. For a logical weight matrix $\mathbf{W}$ of shape $(N, K)$ and an activation matrix $\mathbf{X}$ of shape $(M, K)$, both restricted to the macro's integer value ranges, the result is

$$\mathbf{Y} = \mathbf{X}\,\mathbf{W}^{\!\top}, \qquad Y_{m,n} = \sum_{k} X_{m,k}\,W_{n,k},$$

returned as an integer pre-requantize tensor. This integer $\mathbf{Y}$ is the pre-ADC ideal: it is the value the decomposition reconstructs exactly. The realized result carries the per-tile ADC quantization of each constituent read (the deviation enters only through the analog tiles, never the slicing or aggregation arithmetic). Bias addition and the requantization back to the activation grid are not part of the macro; they belong to the operator that wraps it.

The macro has two orthogonal axes that together place this matmul on physical tiles: a matrix-**tiling** axis ($T_r$, $T_c$) that splits any matmul too large for one tile, and a precision-**slicing** axis ($S_w$, $S_a$, specific to compute-in-memory) that decomposes a high-precision value into tile-carriable pieces.

### Matrix tiling (T_r, T_c)

A logical weight matrix wider or taller than one tile is split into a grid of tiles: $T_r = \lceil N / N_{\mathrm{col}} \rceil$ tiles along the output axis (rows of the transposed weight) and $T_c = \lceil K / N_{\mathrm{row}} \rceil$ tiles along the contraction axis. The per-tile reads on the $T_c$ contraction tiles are summed back into one dot product. This axis is application-neutral — it applies to any matmul — and adds no value decomposition.

### Precision slicing (S_w, S_a)

A **value** (level 2, role-neutral: a weight on the weight side, an activation on the input side) whose range exceeds what one tile cell can carry is decomposed into positional **slices** (level 1). A weight is cut into $S_w$ slices and an activation into $S_a$ slices, each slice a fixed-capacity piece of $D$ **digits** (level 0, the integer symbol one xbar cell carries at digit radix $r$). The per-slice positional weight is the **slice radix** $R = r^{D}$, and the LSB-first slice weights are $(1, R, R^{2}, \dots)$, so the value recombines from its slices $m_i$ by the radix-weighted shift-add

$$M = \sum_{i} m_i\, R^{i}.$$

The per-slice value range is computed from the digit count $D$ and the digit radix $r$ published by the [physical-tile contract](../xbar/base.md) (the xbar is the authority for the digit/slice interface). The slice counts $S_w$, $S_a$ are config-given, not inferred. A degenerate member with $S_w = S_a = 1$ performs no slicing.

### Decompose $\leftrightarrow$ aggregate

Slicing a value into positional slices and recombining the per-tile partial reads are inverse operations: the macro decomposes the value into slices, and the radix-weighted shift-add that reconstructs $M$ is the aggregation primitive specified in [digital/shift_adder](../digital/shift_adder.md). The shift-adder folds the slice axis with the positional weights $(1, R, R^2, \dots)$; the contraction-tile sum over $T_c$ is a plain integer accumulation.

## Noise & non-idealities

N/A at the contract level — the macro is a value-domain organizer and adds no analog non-ideality of its own. Every non-ideality enters through the tiles it reads (IR drop, device noise, finite-gain clamps, ADC quantization) and through the digital reduction blocks (exact by construction). The tile-level sources are specified in the physical-tile contract in [xbar/base](../xbar/base.md) and the topology families beneath it.

## Parameters

The abstract contract has no parameters of its own; a concrete member's parameters are its tile configuration plus its slice counts. The architectural value-domain capabilities a macro publishes are below; per-mode parameter tables are in the family documents.

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `w_value_range` | inclusive integer logical-weight range accepted | — | Design |
| `x_value_range` | inclusive integer logical-activation range accepted | — | Design |
| `adc_mode_num` | number of supported ADC operating points | — | Design |
| `adc_max_bits` | maximum ADC resolution across operating points | — | Design |

Provenance terms are defined in [parameter_provenance](../parameter_provenance.md). Runtime inputs (the weight and activation tensors, the ADC operating point) are inputs, not parameters.

### ADC operating-point surface

The macro inherits a discrete ADC operating-point surface from the tiles it reads. It publishes the number of supported operating points `adc_mode_num` (valid `adc_mode` indices are $[0, \mathrm{adc\_mode\_num})$) and the maximum resolution `adc_max_bits`, and for any operating point it publishes the recovery-side rescale factor $s$ relating the integer dot product to the digitized code,

$$M_{\mathrm{ideal}} \approx \mathrm{code}\cdot s.$$

The macro derives this surface from its tiles; the rescale-factor convention and its calibration are the physical-tile contract in [xbar/base](../xbar/base.md#output-rescale). A degenerate member that performs an exact integer matmul publishes a sentinel surface (`adc_mode_num = 1`, `adc_max_bits = 0`, $s = 1$), where `adc_max_bits = 0` signals "no output quantization".

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $\mathbf{W}$ | logical weight matrix (runtime input) | — | `weight` |
| $\mathbf{X}$ | logical activation matrix (runtime input) | — | `input` |
| $\mathbf{Y}$ | pre-requantize integer output | — | `matmul` return |
| $N, K, M$ | output, contraction, and activation-row dims | — | `w_logical_shape`, input shape |
| $S_w, S_a$ | weight-, activation-slice counts (precision-slicing axis) | — | `w_slice_num`, `x_slice_num` |
| $T_r, T_c$ | output-, contraction-axis tile counts (matrix-tiling axis) | — | — |
| $D$ | digits per slice (from the tile) | — | `w_digit_count` |
| $r$ | digit radix (from the tile) | — | `w_digit_radix` |
| $R$ | slice radix, $R = r^{D}$ | — | `slice_radix` |
| $m_i$ | value carried by slice $i$ | — | — |
| $s$ | output rescale factor | — | `adc_rescale_factor` |
| $M_{\mathrm{ideal}}$ | ideal integer dot product | — | — |

The precision-slicing counts $S_w$, $S_a$ and the matrix-tiling counts $T_r$, $T_c$ are the two axes defined in the governing equations above; their value-domain folds enter the per-mode equations of the family documents. The per-slice value range is computed from the digit count $D$ and digit radix $r$ published by the physical tile (see [xbar/base](../xbar/base.md)); the integer value ranges the macro accepts are published as `w_value_range` and `x_value_range`.

## Assumptions, scope & validity

Stated assumptions of the macro contract:

- The macro returns a pre-requantize integer result; bias and requantization are the consuming operator's responsibility, not the macro's.
- Value ranges are a published capability, not an enforced bound: a member trusts the caller to supply integer weights and activations already inside `w_value_range` and `x_value_range`. The per-slice value range is computed from the tile's digit count and radix (the xbar interface is the authority); the slice counts $S_w$, $S_a$ are config-given, not inferred. Out-of-range inputs are not checked and produce undefined results.
- The decomposition is value-domain exact: the only deviation from the exact integer dot product is the analog non-ideality of the constituent tile reads, not the slicing or aggregation arithmetic.

TODO (domain author): state the validity boundary of the slice-and-shift-add decomposition — the exact per-slice value range per encoding, the saturation of the positional recombination $M = \sum_i m_i R^i$, the largest dot-product magnitude representable before the ADC code clamps, and any regime where the value-domain-exact assumption breaks.

## Validation

TODO: link [validation/macro](../../validation/README.md) — agreement of each mode against the ideal twin under a noise-off policy, and bit-exactness of the decompose $\leftrightarrow$ aggregate dual.

## References

TODO: cite the bit-sliced compute-in-memory architecture and the positional shift-add recombination.

---

- **Internals**: [macro base internals](../../internals/macro/base.md)
- **Validation**: TODO — `validation/macro` (not yet written)
- **Configuration**: [config reference](../../api/README.md)
- **Decisions**: N/A — no ADR governs this module.
