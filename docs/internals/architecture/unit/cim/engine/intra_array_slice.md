# IntraArraySliceCimEngine

The intra-macro `Sw` variant places weight slices on adjacent logical output
ports and owns reducers for input phase, `Sa`, `Sw`, and `Tc`.

## Design decisions

- **`N` is padded to a multiple of
  `weights_per_macro = output_num // Sw`.** No weight straddles two macros.
- **Idle output ports are zero-padded and trimmed before `Sw` reconstruction.**

## Contracts & invariants

- **Construction guard.** `w_slice_num <= output_num`.
- **Organized W shape** is
  `[..., M=1, Sa=1, Tc, Tr, input_num, output_num]`; **organized X shape** is
  `[..., M, Sa, Tc, Tr=1, input_num]`.
- **Aggregate order** is input phase, adjacent-port `Sw`, `Sa`, `Tc`, then
  flatten and trim.
- **Reducer radices.** `Sa` uses `x_slicer.slice_radix`, `Sw` uses `w_slicer.slice_radix`.

## Performance & resources

The tile's `inst_shape` carries `(M=1, Sa=1, Tc, Tr)` with no `Sw` multiplicity, so peak memory and tile-read work carry no `Sw` factor.

## Gotchas

- **The `Sw` shift-add must run at `dim=-1`** on the `unflatten`ed `(weights_per_macro, Sw)` block; a wrong dim silently reduces the wrong axis (`dim=-4` hits the `Tc` tile-grid axis). See [base](base.md) for the organize/aggregate pairing rule.

---

- **Reference**: [intra_array_slice engine](../../../../../reference/architecture/unit/cim/engine/intra_array_slice.md)
- **Implementation**: `neurox/architecture/unit/cim/engine/intra_array_slice.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`
