# Intra-array slice macro

## Summary

`IntraArraySliceXbarMacro` realizes a sliced matmul by gathering all slices of one weight inside a single tile (Strategy 2): a weight's $S_w$ slices occupy adjacent columns of the same tile, and they are recombined by an intra-tile positional shift-add. It is the dual of [inter_array_slice](inter_array_slice.md) — the same value-domain reach, but packed into one tile's columns rather than spread across planes. This document specifies the value-domain mapping and the aggregation; the shape pipeline is in [internals/macro/xbar/intra_array_slice](../../../internals/macro/xbar/intra_array_slice.md).

## Physical model

A weight value is sliced into $S_w$ slices and an input value into $S_a$ per-cycle slices, as in the inter-array mode. Here the $S_w$ axis is folded *into* the tile's value (column) axis: one tile holds $\lfloor N_{\mathrm{col}} / S_w \rfloor$ whole weights, each weight's $S_w$ slices occupying adjacent columns. The used capacity is $\lfloor N_{\mathrm{col}}/S_w\rfloor \cdot S_w$ columns; the remaining $N_{\mathrm{col}} - \lfloor N_{\mathrm{col}}/S_w\rfloor\cdot S_w$ columns of each tile are idle and zero-padded. No weight straddles two tiles, so the output axis tiles as $T_r = \lceil N / \lfloor N_{\mathrm{col}}/S_w\rfloor\rceil$. There is no separate $S_w$ plane axis — the slices live in the column axis.

## Governing equations

The recombination is the same positional double shift-add as the inter-array mode; only the locus of the weight-slice fold differs. With per-slice weight radix $R_w$, per-cycle activation radix $R_a$, and $P_{m,n}^{(a,w,t)}$ the partial read for activation slice $a$, the column group carrying weight slice $w$, and contraction tile $t$,

$$Y_{m,n} = \sum_{t=0}^{T_c-1} \sum_{w=0}^{S_w-1} R_w^{\,w} \left( \sum_{a=0}^{S_a-1} R_a^{\,a}\, P_{m,n}^{(a,w,t)} \right).$$

The reduction order is fixed: the activation-slice shift-add (innermost, intra-cycle, serial) folds first, then the stride-$S_w$ weight-slice shift-add within the tile's column output (intra-tile), then the plain contraction-tile accumulation over $T_c$, leaving the per-output-tile results to concatenate and trim to $N$.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $S_w$ | per-weight slice count | — | `w_slice_num` |
| $S_a$ | per-activation slice count | — | `x_slice_num` |
| $R_w$ | per-slice weight radix | — | `w_slicer.slice_radix` |
| $R_a$ | per-cycle activation radix | — | `x_slicer.slice_radix` |
| $N_{\mathrm{col}}$ | tile column count | — | `xbar.col_num` |
| $\lfloor N_{\mathrm{col}}/S_w\rfloor$ | per-tile weight capacity | — | structure count |
| $T_c$ | contraction-axis tile count | — | structure count |

The logical dims, value-domain symbols, and the ADC surface are in [macro/base](../base.md#symbols).

## Noise & non-idealities

N/A at the mode level. ADC quantization and analog non-idealities enter through the tile reads, specified in [xbar/base](../../xbar/base.md); the stride-$S_w$ shift-add, the activation-slice shift-add, and the contraction accumulation are exact integer arithmetic.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `xbar_config` | owned physical-tile configuration | — | Design |
| `w_slice_num` ($S_w$) | per-weight slice count | — | Design |
| `x_slice_num` ($S_a$) | per-activation slice count | — | Design |
| `w_encoding` | weight encoding (integer-to-digit-string codec; signed-digit only for canonical) | — | Design |
| `col_accumulator_config` | contraction-tile ($T_c$) accumulator | — | Design |
| `sa_shift_adder_config` | activation-slice ($S_a$) shift-adder | — | Design |
| `sw_shift_adder_config` | weight-slice ($S_w$) intra-tile shift-adder | — | Design |

Activations are unsigned true-form by definition (no activation encoding). Provenance terms: [module_parameter](../../../conventions/module_parameter.md); file-level schema: [config reference](../../../api/README.md).

## Assumptions, scope & validity

- The slice count must not exceed the tile's column count ($S_w \le N_{\mathrm{col}}$); otherwise the per-tile weight capacity is zero and the mode is invalid.
- Weight and input values must fit the slicers' value ranges; the mode does not enforce the range.
- The mode trades column-capacity utilization (the idle padded columns) for fewer tile reads than the [inter_array_slice](inter_array_slice.md) plane stack.

TODO (domain author): the exact slicer value range per encoding and the saturation boundary of the positional recombination.

## Validation

TODO: link [validation/macro](../../../validation/README.md) — agreement against the ideal twin and the stride-$S_w$ shift-add reconstruction.

## References

TODO.

---

- **Internals**: [intra_array_slice internals](../../../internals/macro/xbar/intra_array_slice.md)
- **Validation**: TODO — `validation/macro` (not yet written)
- **Configuration**: [config reference](../../../api/README.md)
- **Decisions**: N/A — no ADR governs this module.
