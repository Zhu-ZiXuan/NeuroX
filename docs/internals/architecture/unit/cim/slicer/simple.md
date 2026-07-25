# Simple slicer

## Design decisions

- **One slice is one macro-level logical value.** `slice_value_range` describes
  what one macro can accept. The slicer infers the positional radix compatible
  with that carrier range and emits one value per slice.
- **Macro-internal encoding remains hidden.** The slicer never reads or emits
  the macro's physical digit representation.

## Contracts & invariants

- `slice_weights` is the least-significant-first sequence over `slice_radix`.
- `slice(x)` returns shape `x.shape + (slice_num,)`.
- Symmetric carrier ranges `[-(R-1), R-1]` imply slice radix $R$. Unsigned
  `[0, R-1]` carriers use true-form encoding and publish `[0, R^S-1]`.
- The value range is a caller contract; this member does not clamp or reject an
  out-of-range input.

## Performance & resources

The value path performs one transcoder pass.

---

- **Reference**: [slice decomposition](../../../../../reference/architecture/unit/family.md)
- **Implementation**: `neurox/architecture/unit/cim/slicer/simple.py`
- **Tests**: `tests/architecture/unit/test_slicer.py`
