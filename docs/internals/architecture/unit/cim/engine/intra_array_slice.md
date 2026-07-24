# IntraArraySliceCimEngine

The intra-tile `Sw` variant. It owns the tile (`CimMacro`), a `SimpleSlicer` and `SerialSlicer`, a sub-phase `Accumulator`, a `Tc` `Accumulator`, and separate `Sa` and `Sw` `ShiftAdder`s.

## Design decisions

- **`N` padded to a multiple of `weights_per_macro`.** `_organize_w` pads `N` up to a multiple of `weights_per_macro = col_num // Sw` before unflattening into row-tiles, keeping each weight's `Sw` slices tile-local so the stride-`Sw` reduction addresses them contiguously.
- **Idle columns padded in organize, trimmed in aggregate.** `_organize_w` zero-pads the `col_num - weights_per_macro * Sw` idle columns; `matmul` trims back to `weights_per_macro * Sw` before the `Sw` unflatten.

## Contracts & invariants

- **Construction guard.** `w_slice_num <= col_num`, else `weights_per_macro == 0` and the variant is invalid (raised in `__init__`).
- **Organized W shape** is `[..., M=1, Sa=1, Tc, Tr, data_num=col_num, D, row_num]` — no `Sw` axis (it is inlined into `data_num`). **Organized X shape** is `[..., M, Sa, Tc, Tr=1, row_num]`.
- **Aggregate axis indices** (post-VMM, shape `[..., P, *w_batch~, M, Sa, Tc, Tr, data_num]` after `_unroll_sub_phase` and the tile read): int64 upcast, sub-phase accumulate at `dim=_sub_phase_dim = -(b+6)` (b = len(w_batch); the `phase_accumulator`, inst shape `(w_parallel, Tc, Tr)` — per tile output port) → `[..., M, Sa, Tc, Tr, data_num]`, trim idle to `weights_per_macro*Sw`, `unflatten(data_num → weights_per_macro, Sw)`, `Sw` shift-add at `dim=-1` (intra-tile, stride-`Sw`), `Sa` shift-add at `dim=-4`, `Tc` accumulate at `dim=-3`, flatten `(Tr, weights_per_macro)`, trim to `N`.
- **Reducer radices.** `Sa` uses `x_slicer.slice_radix`, `Sw` uses `w_slicer.slice_radix`.

## Performance & resources

The tile's `inst_shape` carries `(M=1, Sa=1, Tc, Tr)` with no `Sw` multiplicity, so peak memory and tile-read work carry no `Sw` factor.

## Gotchas

- **The `Sw` shift-add must run at `dim=-1`** on the `unflatten`ed `(weights_per_macro, Sw)` block; a wrong dim silently reduces the wrong axis (`dim=-4` hits the `Tc` tile-grid axis). See [base](base.md) for the organize/aggregate pairing rule.

---

- **Reference**: [intra_array_slice engine](../../../../../reference/architecture/unit/cim/engine/intra_array_slice.md)
- **Implementation**: `neurox/architecture/unit/cim/engine/intra_array_slice.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`
