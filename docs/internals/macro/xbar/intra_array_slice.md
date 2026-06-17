# IntraArraySliceXbarMacro — Implementation

## Summary

`IntraArraySliceXbarMacro` (`xbar/intra_array_slice.py`): the intra-tile `Sw` mode. It owns the same children as the inter-array mode — tile, `SimpleSlicer`, `SerialSlicer`, an `Accumulator` and two `ShiftAdder`s — but folds the `Sw` axis into the tile column axis. Spec: [reference/macro/xbar/intra_array_slice](../../../reference/macro/xbar/intra_array_slice.md).

## Design decisions

- **`Sw` folded into the column axis at organize time.** A weight's `Sw` slices are flattened into adjacent columns (`flatten(weights_per_xbar, Sw)`), so the organized tensor has *no* `Sw` leading axis — the slices live inside `data_num`. This is the dual of the [inter_array_slice](inter_array_slice.md) plane stack and is what makes the two modes' aggregates incompatible.
- **`N` padded so no weight straddles two tiles.** Before unflattening into row-tiles, `N` is padded up to a multiple of `weights_per_xbar = col_num // Sw`. Whole logical weights stay tile-local, which is required for the intra-tile stride-`Sw` reduction to address one weight's slices contiguously.
- **Idle columns explicitly padded.** The used capacity is `weights_per_xbar * Sw`; the remaining `col_num - weights_per_xbar * Sw` columns are zero-padded and dropped before the `Sw` unflatten in the aggregate.

## Contracts & invariants

- **Construction guard.** `w_slice_num <= col_num`, else `weights_per_xbar == 0` and the mode is invalid (raised in `__init__`).
- **Organized W shape** is `[..., M=1, Sa=1, Tc, Tr, data_num=col_num, D, row_num]` — no `Sw` axis (it is inlined into `data_num`). **Organized X shape** is `[..., M, Sa, Tc, Tr=1, row_num]`.
- **Aggregate axis indices** (post-VMM, shape `[..., M, Sa, Tc, Tr, data_num]`): trim idle to `wpx*Sw`, `unflatten(data_num → wpx, Sw)`, `Sw` shift-add at `dim=-1` (intra-tile, stride-`Sw`), `Sa` shift-add at `dim=-4`, `Tc` accumulate at `dim=-3`, flatten `(Tr, wpx)`, trim to `N`. The `Sw` reduction at `dim=-1` is the structural difference from the inter-array mode's `dim=-4`.
- **Reducer radices** as in the inter-array mode: `Sa` uses `x_slicer.slice_radix`, `Sw` uses `w_slicer.slice_radix`.

## Performance & resources

Trades column utilization (idle padded columns) for fewer tile instances than the inter-array plane stack: the tile's `inst_shape` carries `(M=1, Sa=1, Tc, Tr)` with no `Sw` multiplicity. The dominant cost is the tile read; `matmul` runs eager (the DC solve compiles as a separate regional leaf).

## Gotchas

- **The `Sw` fold is in the data axis — reducing the wrong dim is silent.** The intra-tile `Sw` shift-add must run at `dim=-1` on the `unflatten`ed `(wpx, Sw)` block; applying the inter-array `dim=-4` here reduces a tile-grid axis instead and produces a wrong-but-plausible shape. The organize/aggregate pair is mode-specific by design — see [base](base.md).
- **`weights_per_xbar` floors.** With `Sw` not dividing `col_num`, some columns are permanently idle; this is intended capacity loss, not a bug.

## Known limitations

- N/A.

---

- **Reference**: [intra_array_slice](../../../reference/macro/xbar/intra_array_slice.md)
- **Implementation**: `neurox/macro/xbar/intra_array_slice.py`
- **Tests**: `tests/test_xbar_macro.py`
- **Decisions**: N/A — no ADR governs this module.
