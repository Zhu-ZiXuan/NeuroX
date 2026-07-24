# Intra-array slice engine

A sliced integer matmul that packs all $S_w$ slices of one weight into adjacent columns of a single tile, recombining them by an intra-tile positional shift-add.

## Physical model

A weight value is sliced into $S_w$ slices and an input value into $S_a$ per-cycle slices. The $S_w$ axis is folded *into* the tile's value (column) axis: one tile holds $\lfloor N_{\mathrm{col}} / S_w \rfloor$ whole weights, each weight's $S_w$ slices occupying adjacent columns. The used capacity is $\lfloor N_{\mathrm{col}}/S_w\rfloor \cdot S_w$ columns; the remaining $N_{\mathrm{col}} - \lfloor N_{\mathrm{col}}/S_w\rfloor\cdot S_w$ columns of each tile are idle. No weight straddles two tiles, so the output axis tiles as $T_r = \lceil N / \lfloor N_{\mathrm{col}}/S_w\rfloor\rceil$. There is no separate $S_w$ plane axis — the slices live in the column axis.

## Governing equations

The recombination is a positional double shift-add. With per-slice weight radix $R_w$, per-cycle activation radix $R_a$, and $P_{m,n}^{(a,w,t)}$ the partial read for activation slice $a$, the column group carrying weight slice $w$, and contraction tile $t$,

$$Y_{m,n} = \sum_{t=0}^{T_c-1} \sum_{w=0}^{S_w-1} R_w^{\,w} \left( \sum_{a=0}^{S_a-1} R_a^{\,a}\, P_{m,n}^{(a,w,t)} \right).$$

Each partial read $P_{m,n}^{(a,w,t)}$ is itself the sub-phase accumulation of the tile's per-plane codes ([engine family](family.md)). The reduction order is fixed: the sub-phase accumulate folds first, then the activation-slice shift-add (intra-cycle, serial), then the stride-$S_w$ weight-slice shift-add within the tile's column output (intra-tile), then the plain contraction-tile accumulation over $T_c$, leaving the per-output-tile results to concatenate and trim to $N$.

## Noise & non-idealities

N/A at the variant level. ADC quantization and analog non-idealities enter through the tile reads, specified by the [CIM macro family](../../../../primitive/macro/cim/family.md); the stride-$S_w$ shift-add, the activation-slice shift-add, and the contraction accumulation are exact integer arithmetic.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `cim_macro_config` | owned physical-tile configuration | — | — | Design |
| `w_slice_num` ($S_w$) | per-weight slice count | — | $1 \le S_w \le N_{\mathrm{col}}$ | Design |
| `x_slice_num` ($S_a$) | per-activation slice count | — | $S_a \ge 1$ | Design |
| `w_encoding` | weight encoding (integer-to-digit-string codec; signed-digit only for canonical) | — | — | Design |
| `phase_accumulator_config` | sub-phase-axis per-tile-port accumulator | — | — | Design |
| `col_accumulator_config` | contraction-tile ($T_c$) accumulator | — | — | Design |
| `sa_shift_adder_config` | activation-slice ($S_a$) shift-adder | — | — | Design |
| `sw_shift_adder_config` | weight-slice ($S_w$) intra-tile shift-adder | — | — | Design |

Activations are unsigned true-form by definition (no activation encoding). Provenance terms: [module_parameter](../../../../../conventions/module_parameter.md); file-level schema: [config reference](../../../../../api/README.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $S_w$ | per-weight slice count | — | `w_slice_num` |
| $S_a$ | per-activation slice count | — | `x_slice_num` |
| $R_w$ | per-slice weight radix | — | `w_slicer.slice_radix` |
| $R_a$ | per-cycle activation radix | — | `x_slicer.slice_radix` |
| $N_{\mathrm{col}}$ | tile column count | — | `cim_macro.col_num` |
| $\lfloor N_{\mathrm{col}}/S_w\rfloor$ | per-tile weight capacity | — | structure count |
| $T_c$ | contraction-axis tile count | — | structure count |
| $T_r$ | output-axis tile count | — | structure count |

The logical dims, value-domain symbols, and the ADC surface are in [unit/family](../../family.md#symbols).

## Assumptions, scope & validity

- The slice count must not exceed the tile's column count ($S_w \le N_{\mathrm{col}}$); otherwise the per-tile weight capacity is zero and the variant is invalid.
- Weight and input values must fit the slicers' value ranges; the variant does not enforce the range.
- The variant trades column-capacity utilization (the idle columns) for fewer tile reads: the tile grid carries no $S_w$ plane multiplicity.

TODO (domain author): the exact slicer value range per encoding and the saturation boundary of the positional recombination.

## Validation

TODO: add validation evidence for agreement against the ideal twin and the stride-$S_w$ shift-add reconstruction.

## References

TODO.

---

- **Internals**: [intra_array_slice engine internals](../../../../../internals/architecture/unit/cim/engine/intra_array_slice.md)
- **Validation**: TODO — `validation/macro` (not yet written)
- **Configuration**: [config reference](../../../../../api/README.md)
