# Intra-Array Slice Macro

## Summary

`IntraArraySliceXbarMacro` realizes a sliced matmul by gathering all slices of one logical weight inside a single tile (Strategy 2): a weight's $S_w$ slices occupy adjacent columns of the same tile, and they are recombined by an intra-tile positional shift-add. It is the dual of [inter_array_slice](inter_array_slice.md) — the same value-domain reach, but packed into one tile's columns rather than spread across planes. This document specifies the value-domain mapping and the aggregation; the shape pipeline is in [internals/macro/xbar/intra_array_slice](../../../internals/macro/xbar/intra_array_slice.md).

## Physical model

A logical weight is sliced into $S_w$ per-weight slices and an activation into $S_a$ per-cycle slices, as in the inter-array mode. Here the $S_w$ axis is folded *into* the tile's data (column) axis: one tile holds $\lfloor N_{\mathrm{col}} / S_w \rfloor$ whole logical weights, each weight's $S_w$ slices occupying adjacent columns. The used capacity is $\lfloor N_{\mathrm{col}}/S_w\rfloor \cdot S_w$ columns; the remaining $N_{\mathrm{col}} - \lfloor N_{\mathrm{col}}/S_w\rfloor\cdot S_w$ columns of each tile are idle and zero-padded. No logical weight straddles two tiles, so the output axis tiles as $T_r = \lceil N / \lfloor N_{\mathrm{col}}/S_w\rfloor\rceil$. There is no separate $S_w$ plane axis — the slices live in the column axis.

## Governing equations

The recombination is the same positional double shift-add as the inter-array mode; only the locus of the weight-slice fold differs. With per-slice weight radix $r_w$, per-cycle activation radix $r_a$, and $P_{m,n}^{(a,w)}$ the partial read for activation slice $a$ and the column group carrying weight slice $w$,

$$Y_{m,n} = \sum_{w=0}^{S_w-1} r_w^{\,w} \left( \sum_{a=0}^{S_a-1} r_a^{\,a}\, P_{m,n}^{(a,w)} \right).$$

Because the $S_w$ slices share one tile read, the weight-slice shift-add is a stride-$S_w$ reduction within the tile's column output (intra-tile), performed before the activation-slice shift-add; the plain contraction-tile accumulation then folds $T_c$, and the per-tile weight results concatenate and trim to $N$.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $S_w$ | per-weight slice count | — | `w_slice_num` |
| $S_a$ | per-activation slice count | — | `x_slice_num` |
| $r_w$ | per-slice weight radix | — | `w_slicer.slice_radix` |
| $r_a$ | per-cycle activation radix | — | `x_slicer.slice_radix` |
| $N_{\mathrm{col}}$ | tile column count | — | `xbar.col_num` |

Per-tile weight capacity ($\lfloor N_{\mathrm{col}}/S_w\rfloor$), tile-grid counts, logical dims, value-domain symbols, and the ADC surface are referred to by code field or specified in [macro/base](../base.md#symbols).

## Noise & non-idealities

N/A at the mode level. ADC quantization and analog non-idealities enter through the tile reads, specified in [xbar/base](../../xbar/base.md); the stride-$S_w$ shift-add, the activation-slice shift-add, and the contraction accumulation are exact integer arithmetic.

## Parameters

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `xbar_config` | owned physical-tile configuration | — | Design |
| `w_slice_num` ($S_w$) | per-weight slice count | — | Design |
| `x_slice_num` ($S_a$) | per-activation slice count | — | Design |
| `w_encoding` | signed-digit encoding for the weight slicer | — | Design |
| `col_accumulator_config` | contraction-tile ($T_c$) accumulator | — | Design |
| `sa_shift_adder_config` | activation-slice ($S_a$) shift-adder | — | Design |
| `sw_shift_adder_config` | weight-slice ($S_w$) intra-tile shift-adder | — | Design |

Activations are unsigned true-form by definition (no activation encoding). Provenance terms: [parameter_provenance](../../parameter_provenance.md); file-level schema: [config reference](../../../api/README.md).

## Assumptions, scope & validity

- The slice count must not exceed the tile's column count ($S_w \le N_{\mathrm{col}}$); otherwise the per-tile weight capacity is zero and the mode is invalid.
- Logical weights and activations must fit the slicers' value ranges; the mode does not enforce the range.
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
