# Xbar Value-Domain Primitives

This directory documents the value-domain mapping primitives shared by every xbar macro mode.

## Public surface

- `Slicer` ABC — decomposes integers into `[..., slice_num, digit_num]`. The interface is intentionally narrow: a `slice(x)` method returning the digit tensor, plus three `@property` accessors `value_range`, `slice_radix`, `slice_weights` (LSB-first positional weights of the slice axis).
  - `SerialSlicer` — `Sa` path; `digit_num == 1`.
  - `SimpleSlicer` — `Sw` path; direct digitise-then-group.

For macro-level geometric chunking (tile), use the chunk-and-pad staticmethod on the xbar macro abstract base — see [`docs/dev/architecture/xbar_macro.md`](../../../architecture/xbar_macro.md).
