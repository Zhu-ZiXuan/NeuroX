# Mapping Architecture

This document records the value-domain primitives that every xbar macro mode reuses. Higher-level macro structure (organize / aggregate / mode subclass) lives in [`xbar_macro.md`](xbar_macro.md).

## Layered split

- **Transcoder** — pure integer ↔ digit-list math. One concrete subclass per encoding policy (true-form, radix-complement, canonical signed-digit), all built from the same `Transcoder` ABC.
- **Slicer** — decomposes an integer tensor into a trailing `[..., slice_num, digit_num]` pair. The activation path uses a slicer with `digit_num == 1` and a hard-coded unsigned true-form encoding; the weight path uses a direct digitise-then-group slicer whose encoding is configurable.

## Slicer contract

The `Slicer` ABC is a pure interface. Its externally observable surface is exactly:

- `slice(x: Tensor) -> Tensor` — decompose ``x``; the return carries trailing-2 axes `[slice_num, digit_num]`.
- `value_range: tuple[int, int]` `@property` — algorithm-side integer range one input scalar can take.
- `slice_radix: int` `@property` — per-slice positional radix; drives the downstream shift-add reduction.
- `slice_weights: tuple[int, ...]` `@property` — LSB-first per-slice positional weights `(1, R, R², …)`. Names how the slice axis recombines; consumers materialise it via `torch.tensor(slicer.slice_weights, dtype=..., device=...)` at their own boundary.

Concrete subclasses take only the constructor parameters they actually use. Structural defaults (e.g. `SerialSlicer`'s `digit_count == 1` and unsigned digit grid) are internal — the slicer does not accept and then re-validate parameters that callers would have to fabricate. The xbar macro reads `w_digit_count`, `w_digit_radix`, and `x_range` from its xbar and forwards exactly the values each slicer subclass needs.

## Encoding policy as a string discriminator

Configs and TOML carry the encoding choice as the string discriminator `"true_form" | "complement" | "canonical"`. Materialise the corresponding subclass through `Transcoder.create(encoding, *, radix, digit_num)`. The activation-side slicer takes no encoding parameter — its primitive grid is unsigned, so the encoding is fixed to true-form.

## Naming convention

- Slicer / transcoder / macro APIs use `value_range` for the algorithm-side complete-value range.
- Xbar uses `digit_range` for the primitive single-cell range.

This keeps the algorithm-side range distinct from the primitive-cell range.
