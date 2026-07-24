# Direct slicer

## Design decisions

- **Identity decomposition.** `DirectSlicer` validates the configured inclusive
  value range and appends two structural singleton axes without changing values
  or dtype.
- **Alphabet size as radix.** With one slice, no positional ratio is applied.
  `slice_radix` therefore reports the configured alphabet size.

## Contracts & invariants

- `slice(x)` returns shape `x.shape + (1, 1)` and preserves values and dtype.
- Every input value must lie inside `value_range`; an out-of-range value raises
  `ValueError`.
- `slice_weights` is `(1,)`.

---

- **Reference**: [slice decomposition](../../../../../reference/architecture/unit/family.md)
- **Implementation**: `neurox/architecture/unit/cim/slicer/direct.py`
- **Tests**: `tests/architecture/unit/test_slicer.py`
