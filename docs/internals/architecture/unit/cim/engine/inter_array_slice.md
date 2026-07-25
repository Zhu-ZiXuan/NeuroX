# InterArraySliceCimEngine

The cross-plane `Sw` variant owns a macro, weight and activation slicers, and
four reducers for input phase, `Sa`, `Sw`, and `Tc`.

## Design decisions

- **`Sw` hoisted to a leading tile axis.** The organize step permutes the per-weight slice axis into the canonical `[Sa, Sw, Tc, Tr]` position. The tile's `inst_shape` carries the full `(M=1, Sa=1, Sw, Tc, Tr)` prefix — the simulator batches the `Sw` planes into one tensor for GPU throughput.
- **Aggregate is the dual of organize.** Because `Sw` sits outside the tile data axis, its shift-add reduces over the leading `Sw` axis rather than within the tile — the aggregate-side mirror of the organize-side hoist.

## Contracts & invariants

- **Organized W shape** is
  `[..., M=1, Sa=1, Sw, Tc, Tr, input_num, output_num]`; **organized X
  shape** is `[..., M, Sa, Sw=1, Tc, Tr=1, input_num]`.
- **Aggregate order** is input phase, `Sa`, `Sw`, `Tc`, then flatten
  `(Tr, output_num)` and trim.
- **Reducer radices.** The `Sa` shift-add takes `x_slicer.slice_radix`, the `Sw` shift-add `w_slicer.slice_radix`; the pairing is load-bearing — swapping the two breaks the recombination.

## Performance & resources

Tile work scales with the materialized `Sw` plane count: the batched tile tensor carries the full `Sw × Tc × Tr` instance multiplicity, so peak memory grows with `Sw`. The dominant cost is the batched tile read; `matmul` runs eager (the DC solve compiles as a separate regional leaf).

## Gotchas

- **`Sw` planes are batched, not separate calls.** The "logically uses $S_w \times \dots$ tiles" framing is architectural; in the code one `vec_mat_mul` processes all planes at once. Do not assume per-plane streaming.

---

- **Reference**: [inter_array_slice engine](../../../../../reference/architecture/unit/cim/engine/inter_array_slice.md)
- **Implementation**: `neurox/architecture/unit/cim/engine/inter_array_slice.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`
