# Slicer Family

Slicers handle **value decomposition** only — they take a high-precision integer-valued tensor and produce a digit tensor with trailing shape `[..., slice_num, digit_num]`. Geometric tiling is each xbar-macro mode's job; slicers are encoding-agnostic about tile shape.

## Generic interface

Every slicer accepts the same triple from the surrounding mapper:

- `digit_count` — number of digits per slice.
- `digit_radix` — positional base of the in-slice digit combination.
- `digit_range` — inclusive integer range a single primitive cell can carry.

`value_range` and `slice_radix` are slicer **outputs**, not inputs.

## Concrete implementations

### `SimpleSlicer` — direct digitise-then-group

Use case: weight decomposer. One scalar becomes `slice_num · digit_count` radix-`digit_radix` digits in one encoding pass; consecutive `digit_count` digits are grouped LSB-first into one xbar-word slice.

Encoding policy is selected at construction (`encoding ∈ {true_form, complement, canonical}`) and forwarded to the transcoder; digit-grid compatibility with the xbar's primitive cell is a caller-side invariant — `SimpleSlicer` does not validate it.

Output (uniform `SlicingPlan` contract):

- `values` shape: `[..., slice_num, digit_count]`.
- `slice_weights = [1, R, R², …]` with `R = r^digit_count`.
- `digit_weights = [1, r, r², …, r^(digit_count - 1)]`.
- `value_range` derived from the encoding + total digit count.

### `SerialSlicer` — unsigned radix-`r` positional, `digit_num = 1`

Use case: activation serializer. One unsigned scalar becomes `slice_num` digits at radix `r = len(digit_range)`; each digit drives one xbar input cycle. The structural digit axis is a singleton.

The activation grid is unsigned by construction, and signed-digit encodings can emit negative digits that don't fit an unsigned primitive cell. `SerialSlicer` therefore hard-codes the sign-magnitude (`true_form`) encoding and takes no `encoding` parameter.

Output:

- `values` shape: `[..., slice_num, 1]`.
- `slice_weights = [1, r, r², …, r^(slice_num - 1)]`.
- `digit_weights = [1]`.
- `value_range = (0, r^slice_num - 1)`.

See also:

- `docs/dev/modules/mapper/transcoder/README.md`
- `docs/dev/architecture/mapping.md`
