# Inter-array slice engine

This variant spends one tile plane per weight slice to widen the weight value range beyond a single tile's value domain, recombining the planes by a cross-plane positional shift-add.

## Physical model

A weight value is sliced into $S_w$ slices, each a tile-carriable integer; an input value is sliced into $S_a$ per-cycle slices. In this variant the $S_w$ slice axis is kept *outside* the tile's value axis: slice index $i$ of every weight lives on tile plane $i$, so the architecture is a stack of $S_w$ separate tile planes, each a $T_r \times T_c$ tile grid — $S_w \times T_r \times T_c$ tiles in total. Within a plane, whole weight-slices tile along columns and the contraction tiles along rows.

## Governing equations

The weight slicing is positional with per-slice radix $R_w$ and the activation slicing positional with per-cycle radix $R_a$, so the exact dot product reconstructs by a nested radix-weighted shift-add over both slice axes and a plain sum over the contraction tiles. Writing the per-plane, per-cycle, per-contraction-tile partial read as $P_{m,n}^{(a,w,t)}$ (the raw tile read on activation slice $a$, weight plane $w$, and contraction tile $t$),

$$Y_{m,n} = \sum_{t=0}^{T_c-1} \sum_{w=0}^{S_w-1} R_w^{\,w} \left( \sum_{a=0}^{S_a-1} R_a^{\,a}\, P_{m,n}^{(a,w,t)} \right).$$

Each partial read $P_{m,n}^{(a,w,t)}$ is itself the sub-phase accumulation of the tile's per-plane codes ([engine family](README.md)). The reduction order is fixed: the sub-phase accumulate folds first, then the activation-slice shift-add (intra-cycle, serial), then the weight-slice shift-add (cross-plane, weighted sum), then the plain contraction-tile accumulation, leaving the per-output-tile results to concatenate and trim to $N$.

## Numerical method

N/A — the decomposition and the radix-weighted shift-add recombination are exact integer arithmetic, introducing no numerical scheme.

## Noise & non-idealities

N/A at the variant level. ADC quantization and analog non-idealities enter through the per-plane tile reads, specified in [CimMacro base](../../../../primitive/macro/cim/README.md); both shift-adds and the contraction accumulation are exact integer arithmetic.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `cim_macro_config` | owned physical-tile configuration | — | — | Design |
| `w_slice_num` ($S_w$) | per-weight slice count | — | ≥ 1 | Design |
| `x_slice_num` ($S_a$) | per-activation slice count | — | ≥ 1 | Design |
| `w_encoding` | weight encoding (integer-to-digit-string codec; signed-digit only for canonical) | — | — | Design |
| `phase_accumulator_config` | sub-phase-axis per-tile-port accumulator | — | — | Design |
| `col_accumulator_config` | contraction-tile ($T_c$) accumulator | — | — | Design |
| `sa_shift_adder_config` | activation-slice ($S_a$) shift-adder | — | — | Design |
| `sw_shift_adder_config` | weight-slice ($S_w$) cross-plane shift-adder | — | — | Design |

Activations are unsigned true-form by definition (no activation encoding). Provenance terms: [module_parameter](../../../../../conventions/module_parameter.md); file-level schema: [config reference](../../../../../api/README.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $S_w$ | per-weight slice count | — | `w_slice_num` |
| $S_a$ | per-activation slice count | — | `x_slice_num` |
| $R_w$ | per-slice weight radix | — | `w_slicer.slice_radix` |
| $R_a$ | per-cycle activation radix | — | `x_slicer.slice_radix` |
| $T_c$ | contraction-axis tile count | — | structure count |
| $T_r$ | output-axis tile count | — | structure count |

The equations above use these slice radices and the tile counts $T_r$, $T_c$; the logical dims, value-domain symbols, and ADC surface are in [unit/family](../../family.md#symbols).

## Assumptions, scope & validity

- Weight and input values must fit the slicers' value ranges; the variant does not enforce the range.
- The variant trades tile count for value-domain reach: $S_w$ planes are spent to widen the weight range.

TODO (domain author): the exact slicer value range per encoding and the saturation boundary of the positional recombination.

## Validation

TODO: link [validation/macro](../../../../../validation/README.md) — agreement against the ideal twin and the cross-plane shift-add reconstruction.

## References

TODO.

---

- **Internals**: [inter_array_slice engine internals](../../../../../internals/architecture/unit/cim/engine/inter_array_slice.md)
- **Validation**: TODO — `validation/macro` (not yet written)
- **Configuration**: [config reference](../../../../../api/README.md)
