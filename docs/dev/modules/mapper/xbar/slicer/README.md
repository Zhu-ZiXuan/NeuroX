# Slicer Family

Slicers handle **value decomposition** only — they take a high-precision integer-valued tensor and produce a digit tensor with trailing shape `[..., slice_num, digit_num]`.

Geometry decomposition is the [`tiler`](docs/dev/modules/mapper/xbar/tiler/README.md)'s job; slicers are encoding-agnostic about tile shape.

## Generic interface

Every slicer accepts the same triple from the surrounding mapper:

- `digit_count` — number of digits per slice
- `digit_radix` — positional base of the in-slice digit combination
- `digit_range` — inclusive integer range a single digit can carry

`value_range` is a slicer **output**, not an input — it falls out of the encoding choice (true-form / complement / canonical) combined with the triple above.

## Concrete implementations

### `SimpleSlicer` — direct digitise-then-group

Use case: weight decomposer. One scalar becomes `slice_num · digit_count` radix-`digit_radix` digits in one encoding pass; consecutive `digit_count` digits are grouped LSB-first into one xbar-word slice.

Output (uniform `SlicingResult` contract):

- `values` shape: `[..., slice_num, digit_count]`.
- `slice_weights = [1, R, R^2, ...]` with `R = r^digit_count`.
- `digit_weights = [1, r, r^2, ..., r^(digit_count - 1)]`.
- `value_range` derived from the encoding + total digit count.

### `SerialSlicer` — radix-`r` positional, `digit_num = 1`

Use case: activation serializer. One unsigned scalar becomes `slice_num` digits at radix `r = len(digit_range)`; each digit drives one xbar input cycle. The structural digit axis is a singleton.

Output:

- `values` shape: `[..., slice_num, 1]`.
- `slice_weights = [1, r, r^2, ..., r^(slice_num - 1)]`.
- `digit_weights = [1]`.
- `value_range = (0, r^slice_num - 1)` (trimmed to the unsigned half because activation values are non-negative).

See also:

- `docs/dev/modules/mapper/xbar/base.md`
- `docs/dev/modules/mapper/transcoder.md`
- `docs/dev/architecture/mapping.md`
