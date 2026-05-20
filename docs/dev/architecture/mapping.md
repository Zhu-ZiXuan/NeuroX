# Mapping Architecture

This document records the value-domain primitives that every xbar macro mode reuses. Higher-level macro structure (organize / aggregate / mode subclass) lives in [`xbar_macro.md`](xbar_macro.md).

## Layered split

- **Transcoder** — pure integer ↔ digit-list math. One concrete subclass per encoding policy (true-form, radix-complement, canonical signed-digit), all built from the same `Transcoder` ABC.
- **Slicer** — decomposes an integer tensor into a trailing `[..., slice_num, digit_num]` pair. The activation path uses a slicer with `digit_num == 1` and a hard-coded unsigned true-form encoding; the weight path uses a slice-first-then-digitise slicer whose encoding is configurable.

## Slicer contract

Every slicer returns a `SlicingPlan` dataclass with trailing-2 axes `[slice_num, digit_num]`. The runtime interface is driven by three keyword arguments:

- `digit_count` — xbar-internal digit count per cell.
- `digit_radix` — xbar-internal per-digit radix.
- `digit_range` — primitive cell value range.

`value_range` and `slice_radix` are slicer *outputs*, not inputs.

## Encoding policy as a string discriminator

Configs and TOML carry the encoding choice as the string discriminator `"true_form" | "complement" | "canonical"`. Materialise the corresponding subclass through `Transcoder.create(encoding, *, radix, digit_num)`. The activation-side slicer takes no encoding parameter — its primitive grid is unsigned, so the encoding is fixed to true-form.

## Naming convention

- Slicer / transcoder / macro APIs use `value_range` for the algorithm-side complete-value range.
- Xbar uses `digit_range` for the primitive single-cell range.

This keeps the algorithm-side range distinct from the primitive-cell range.
