# InterArraySliceXbarMacro

## Summary

`InterArraySliceXbarMacro` (`xbar/inter_array_slice.py`): the cross-plane `Sw` mode. It owns a tile, a `SimpleSlicer` (weights), a `SerialSlicer` (activations), and three reducers — an `Accumulator` (`Tc`) plus two `ShiftAdder`s (`Sa`, `Sw`). Spec: [reference/macro/xbar/inter_array_slice](../../../reference/macro/xbar/inter_array_slice.md).

## Design decisions

- **`Sw` hoisted to a leading tile axis.** The organize step permutes the per-weight slice axis into the canonical `[Sa, Sw, Tc, Tr]` position, so one tile plane (`Sw` index) carries one slice of every weight. The tile's `inst_shape` carries the full `(M=1, Sa=1, Sw, Tc, Tr)` prefix — the simulator batches the `Sw` planes as one tensor for GPU throughput, but the architecture is a stack of separate planes.
- **Aggregate is the dual of organize: `Sw` reduces cross-plane.** Because `Sw` sits outside the tile data axis, the `Sw` shift-add is a cross-plane weighted sum over the leading `Sw` axis, distinct from the intra-tile fold of the [intra_array_slice](intra_array_slice.md) mode. The reduction order is `Sa` shift-add → `Sw` shift-add → `Tc` accumulate, fixed by the radix nesting.

## Contracts & invariants

- **Organized W shape** is `[..., M=1, Sa=1, Sw, Tc, Tr, data_num, D, row_num]` (all four leading slice/tile axes present); **organized X shape** is `[..., M, Sa, Sw=1, Tc, Tr=1, row_num]`. The `Sw=1` / `Tr=1` placeholders on X broadcast against the W tensor's real `Sw` / `Tr`.
- **Aggregate axis indices** (post-VMM, shape `[..., M, Sa, Sw, Tc, Tr, data_num]`): `Sa` shift-add at `dim=-5`, `Sw` shift-add at `dim=-4`, `Tc` accumulate at `dim=-3`, then flatten `(Tr, data_num)` and trim to `N`. These indices are valid only because the `[Sa, Sw, Tc, Tr]` order is fixed.
- **Reducer radices.** The `Sa` shift-add uses `x_slicer.slice_radix`, the `Sw` shift-add uses `w_slicer.slice_radix`; both are positional, so the recombination is exact.

## Performance & resources

Tile work scales with the materialized `Sw` plane count: the batched tile tensor carries the full `Sw × Tc × Tr` instance multiplicity, so peak memory grows with `Sw`. The dominant cost is the batched tile read; `matmul` runs eager (the DC solve compiles as a separate regional leaf).

## Gotchas

- **`Sw` planes are batched, not separate calls.** The "logically uses $S_w \times \dots$ tiles" framing is architectural; in the code one `vec_mat_mul` processes all planes at once. Do not assume per-plane streaming.

## Known limitations

- N/A.

---

- **Reference**: [inter_array_slice](../../../reference/macro/xbar/inter_array_slice.md)
- **Implementation**: `neurox/macro/xbar/inter_array_slice.py`
- **Tests**: `tests/test_xbar_macro.py`
- **Decisions**: N/A — no ADR governs this module.
