# IntraArraySliceCimUnit

Owns the tile (`CimMacro`), a `SimpleSlicer` and `SerialSlicer`, a `Tc` `Accumulator`, and separate `Sa` and `Sw` `ShiftAdder`s.

## Design decisions

- **`N` padded to a multiple of `weights_per_xbar`.** `_organize_w` pads `N` up to a multiple of `weights_per_xbar = col_num // Sw` before unflattening into row-tiles, keeping each weight's `Sw` slices tile-local so the stride-`Sw` reduction addresses them contiguously.
- **Idle columns padded in organize, trimmed in aggregate.** `_organize_w` zero-pads the `col_num - weights_per_xbar * Sw` idle columns; `matmul` trims back to `weights_per_xbar * Sw` before the `Sw` unflatten.

## Contracts & invariants

- **Construction guard.** `w_slice_num <= col_num`, else `weights_per_xbar == 0` and the mode is invalid (raised in `__init__`).
- **Organized W shape** is `[..., M=1, Sa=1, Tc, Tr, data_num=col_num, D, row_num]` — no `Sw` axis (it is inlined into `data_num`). **Organized X shape** is `[..., M, Sa, Tc, Tr=1, row_num]`.
- **Aggregate axis indices** (post-VMM, shape `[..., M, Sa, Tc, Tr, data_num]`): trim idle to `wpx*Sw`, `unflatten(data_num → wpx, Sw)`, `Sw` shift-add at `dim=-1` (intra-tile, stride-`Sw`), `Sa` shift-add at `dim=-4`, `Tc` accumulate at `dim=-3`, flatten `(Tr, wpx)`, trim to `N`.
- **Reducer radices.** `Sa` uses `x_slicer.slice_radix`, `Sw` uses `w_slicer.slice_radix`.

## Performance & resources

The tile's `inst_shape` carries `(M=1, Sa=1, Tc, Tr)` with no `Sw` multiplicity, so peak memory and tile-read work carry no `Sw` factor.

## Gotchas

- **The `Sw` shift-add must run at `dim=-1`** on the `unflatten`ed `(wpx, Sw)` block; a wrong dim silently reduces the wrong axis (`dim=-4` hits the `Tc` tile-grid axis). See [base](base.md) for the organize/aggregate pairing rule.

---

- **Reference**: [intra_array_slice](../../../../reference/architecture/unit/cim/intra_array_slice.md)
- **Implementation**: `neurox/architecture/unit/cim/intra_array_slice.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`
