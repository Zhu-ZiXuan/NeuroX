# Serial slicer

## Design decisions

- **One digit per slice.** `SerialSlicer` uses unsigned true-form radix
  decomposition and appends a structural `digit_count = 1` axis.
- **Encoding is fixed.** The constructor accepts only the slice count and digit
  radix because signed encoding is outside this member's unsigned value domain.

## Contracts & invariants

- For slice count $S$ and digit radix $r$, `value_range` is
  $[0,\ r^S-1]$, `slice_radix` is $r$, and `slice_weights` is
  $(1,r,\ldots,r^{S-1})$.
- `slice(x)` returns shape `x.shape + (slice_num, 1)` with the slice axis
  ordered least-significant first.
- The value range is a caller contract; this member does not clamp or reject an
  out-of-range input.

## Performance & resources

The value path performs one transcoder pass followed by one `unsqueeze`.

---

- **Reference**: [slice decomposition](../../../../../reference/architecture/unit/family.md)
- **Implementation**: `neurox/architecture/unit/cim/slicer/serial.py`
- **Tests**: `tests/architecture/unit/test_slicer.py`
