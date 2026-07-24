# Simple slicer

## Design decisions

- **Encode once, group once.** `SimpleSlicer` encodes a value into one digit
  string of length `slice_num * digit_count`, then unflattens that string into
  the trailing `[slice_num, digit_count]` lattice. It does not run one
  transcoder per slice.
- **Encoding selects the complete-value range.** The owned transcoder determines
  `value_range`; the slicer only exposes it and groups the digits.

## Contracts & invariants

- For digit radix $r$ and $D$ digits per slice, `slice_radix` is $r^D$ and
  `slice_weights` is the least-significant-first positional sequence over that
  radix.
- `slice(x)` returns shape `x.shape + (slice_num, digit_count)`.
- The value range is a caller contract; this member does not clamp or reject an
  out-of-range input.

## Performance & resources

The value path performs one transcoder pass followed by one `unflatten`.

---

- **Reference**: [slice decomposition](../../../../../reference/architecture/unit/family.md)
- **Implementation**: `neurox/architecture/unit/cim/slicer/simple.py`
- **Tests**: `tests/architecture/unit/test_slicer.py`
