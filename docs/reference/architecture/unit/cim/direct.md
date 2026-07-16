# Direct unit

The no-slice corner of the precision-slicing axis: one weight slice and one activation slice ($S_w = S_a = 1$). A weight value maps onto one tile's native value range unsliced through the weight codec, and an input value maps onto the tile's input grid directly. The mode applies when the quantization grid already fits inside one tile's value range.

## Physical model

The mode places every weight on the tile array unsliced: each weight occupies one column, and the $K$ contraction inputs occupy the rows. A weight matrix wider than one tile's column count or deeper than its row count is tiled into a grid of tiles — $T_r = \lceil N / N_{\mathrm{col}}\rceil$ tiles along the output axis ($r$ for rows-of-output) and $T_c = \lceil K / N_{\mathrm{row}}\rceil$ tiles along the contraction axis ($c$ for contraction). Each tile holds at most $N_{\mathrm{col}}$ whole weights; the trailing columns of the last output tile and the trailing rows of the last contraction tile are zero-padded. There is no slice axis on either the weight or the activation side, so no positional shift-add recombination is required.

## Governing equations

The mode realizes the exact integer dot product

$$Y_{m,n} = \sum_{k} X_{m,k}\,W_{n,k}$$

as a sum over the contraction-tile grid: each contraction tile computes a partial dot product over its $N_{\mathrm{row}}$ rows, and the partials accumulate (a plain integer sum, no positional weighting) across the $T_c$ tiles to form the full contraction. The output-tile grid carries disjoint columns of $\mathbf{W}^{\!\top}$, so the per-tile results concatenate along the output axis and are trimmed to $N$. The weight enters the tile as a string of $D$ digits in radix $r$ produced by the codec; the tile's primitive read folds the digit axis with the radix-weighted vector, so the digit decomposition is transparent at this level.

## Noise & non-idealities

N/A at the mode level — the mode adds no non-ideality. ADC quantization and analog non-idealities enter through the constituent tile reads, specified in [CimMacro base](../../../primitive/macro/cim/README.md); the cross-tile accumulation is exact integer arithmetic.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `cim_macro_config` | owned physical-tile configuration | — | — | Design |
| `w_encoding` | weight encoding (integer-to-digit-string; signed-digit only for canonical) | — | — | Design |
| `col_accumulator_config` | contraction-tile ($T_c$) accumulator | — | — | Design |

Activations are unsigned true-form by definition, so the mode carries no activation encoding. Provenance terms: [module_parameter](../../../../conventions/module_parameter.md); file-level schema: [config reference](../../../../api/README.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $N_{\mathrm{row}}$ | tile row count | — | `xbar.row_num` |
| $N_{\mathrm{col}}$ | tile column count | — | `xbar.col_num` |
| $T_r$ | output-axis tile count | — | structure count |
| $T_c$ | contraction-axis tile count | — | structure count |
| $D$ | digits per slice (digit count) | — | `xbar.w_digit_count` |
| $r$ | digit radix | — | `xbar.w_digit_radix` |

The logical dims ($N$, $K$, $M$), the value-domain symbols, and the ADC surface are in [unit/family](../family.md#symbols).

## Assumptions, scope & validity

- Weight values must fit the codec's value range (encoding-dependent, e.g. true-form $\pm(r^{D}-1)$) and input values must fit the tile's input grid; neither the mode nor the tile enforces the range, so out-of-range inputs produce undefined results.
- The mode applies only when the quantization grid fits one tile's value range; for $S_w > 1$ or $S_a > 1$ use inter_array_slice or intra_array_slice.

TODO (domain author): the exact value-range bound per encoding and the saturation behaviour at the tile boundary.

## Validation

TODO: link [validation/macro](../../../../validation/README.md) — agreement against the ideal twin in the no-noise limit.

## References

TODO.

---

- **Internals**: [direct internals](../../../../internals/architecture/unit/cim/direct.md)
- **Validation**: TODO — `validation/macro` (not yet written)
- **Configuration**: [config reference](../../../../api/README.md)
