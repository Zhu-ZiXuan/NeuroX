# Slicer Family

Slicers handle **value decomposition** only — they take a high-precision integer-valued tensor and produce a digit tensor with trailing shape `[..., slice_num, digit_num]`. Geometric tiling is each xbar-macro mode's job; slicers are encoding-agnostic about tile shape.

## Generic contract

The `Slicer` ABC is intentionally minimal. The externally observable surface is exactly:

- `slice(x: Tensor) -> Tensor` — decompose ``x``; the return carries trailing-2 axes `[slice_num, digit_num]`.
- `value_range: tuple[int, int]` — `@property`. Inclusive algorithm-side integer range one input scalar can take.
- `slice_radix: int` — `@property`. Per-slice positional radix, used by the downstream shift-add reducer.
- `slice_weights: tuple[int, ...]` — `@property`. LSB-first per-slice positional weights `(1, R, R², …)` with `R = slice_radix`. Part of the contract regardless of immediate consumption — it names how the slice axis recombines.

Concrete subclasses take only the constructor parameters they actually use; the slicer does not re-expose its construction inputs because no caller reads them back through the slicer.

## Concrete implementations

### `SimpleSlicer` — direct digitise-then-group

Use case: weight decomposer. One scalar becomes `slice_num · digit_count` radix-`digit_radix` digits in one encoding pass; consecutive `digit_count` digits are grouped LSB-first into one xbar-word slice.

Constructor parameters:

- `slice_num` — outer slice count (`Sw`).
- `digit_count` — xbar-internal digit count per xbar-word.
- `digit_radix` — xbar-internal per-digit positional radix.
- `encoding` — `"true_form" | "complement" | "canonical"`; signed-digit policy for the unified digit string.

Output:

- `slice(x).shape == x.shape + (slice_num, digit_count)`.
- `slice_radix == digit_radix ** digit_count`.
- `value_range` derived from the encoding + total digit count `slice_num · digit_count`.

Digit-grid compatibility with the xbar's primitive cell is a caller-side invariant.

### `SerialSlicer` — unsigned radix-`r` positional, `digit_num = 1`

Use case: activation serializer. One unsigned scalar becomes `slice_num` digits at radix `digit_radix`; each digit drives one xbar input cycle. The structural digit axis is a singleton.

Constructor parameters:

- `slice_num` — outer slice count (`Sa`).
- `digit_radix` — activation-cell positional radix `r`. The implied unsigned digit range is `[0, r - 1]`.

The encoding is hard-coded to sign-magnitude (`true_form`); other encodings would emit negative digits that don't fit an unsigned primitive cell. `SerialSlicer` therefore takes no `encoding` parameter.

Output:

- `slice(x).shape == x.shape + (slice_num, 1)`.
- `slice_radix == digit_radix`.
- `value_range == (0, digit_radix**slice_num - 1)`.

See also:

- `docs/modules/mapper/transcoder/README.md`
- `docs/dev/architecture/mapping.md`
