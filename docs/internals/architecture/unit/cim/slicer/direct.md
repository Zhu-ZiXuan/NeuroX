# Direct slicer

## Design decisions

- **Identity decomposition.** `DirectSlicer` stores the represented inclusive
  value range and appends two structural singleton axes without scanning or
  changing the runtime values.
- **Alphabet size as radix.** With one slice, no positional ratio is applied.
  `slice_radix` therefore reports the configured alphabet size.

## Contracts & invariants

- `slice(x)` returns shape `x.shape + (1, 1)` and preserves values and dtype.
- `value_range` is planning metadata, not a runtime validation pass.
- `slice_weights` is `(1,)`.

---

- **Reference**: [slice decomposition](../../../../../reference/architecture/unit/family.md)
- **Implementation**: `neurox/architecture/unit/cim/slicer/direct.py`
- **Tests**: `tests/architecture/unit/test_slicer.py`
