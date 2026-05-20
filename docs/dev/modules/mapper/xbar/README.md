# Xbar Value-Domain Primitives

This directory documents the value-domain mapping primitives shared by every xbar macro mode.

## Public surface

- `Slicer` ABC — decomposes integers into `[..., slice_num, digit_num]`.
  - `SerialSlicer` — `Sa` path; `digit_num == 1`.
  - `SimpleSlicer` — `Sw` path; slice-first-then-digitize.
- `SlicingPlan` — frozen dataclass carrying decomposed values, slice / digit positional weights, and the algorithm-side value range.

For macro-level geometric chunking (tile), use the chunk-and-pad staticmethod on the xbar macro abstract base — see [`docs/dev/architecture/xbar_macro.md`](../../../architecture/xbar_macro.md).
