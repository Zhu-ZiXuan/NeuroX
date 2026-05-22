# `DirectXbarMacro`

Transcode-only macro for the `Sw = Sa = 1` case. No slicers, no shift-adders — logical weights and activations map straight onto one xbar's native value range.

## What one xbar holds

One xbar carries every logical weight unsliced. `N` logical weights tile into `Tr = ⌈N / col_num⌉` row-tiles; `K` inputs tile into `Tc = ⌈K / row_num⌉` row-chunks. Per-xbar capacity is `col_num` weights; trailing cols of the last row-tile are zero-padded.

## Organize (W)

`encode` via the weight transcoder → `[..., N, K, D]` → tile `N` along `col_num` → tile `K` along `row_num` → permute → `M=1` placeholder. Final shape: `[..., M=1, Tc, Tr, col_num, D, row_num]` (6 leading dims instead of inter / intra's 7-8 because `Sa` and `Sw` collapse).

## Organize (X)

`chunk_pad` along `K` by `row_num` → insert `Tr=1` placeholder. Final shape: `[..., M, Tc, Tr=1, row_num]`.

## Aggregate

`xbar.vec_mat_mul` → `[..., M, Tc, Tr, col_num]` → `Tc` accumulate → flatten `(Tr, col_num)` → trim to `N`. The macro returns pre-requantize int output; the operator owns bias add and rescale.

No `Sw` / `Sa` shift-adders run because both slice counts are 1.

## Constraints

The macro trusts the upper layer for value ranges:

- Logical weights must fit in `self.w_transcoder.value_range`. With `w_digit_count=D`, `w_digit_radix=r`, the transcoder's value range is encoding-dependent (e.g. true-form: `±(r^D - 1)`).
- Logical activations must fit in `self.xbar.x_range`.

Out-of-range inputs are clipped silently by the xbar; no `assert` is enforced inside the macro.

## When to use

Pick `DirectXbarMacro` when the chosen quantisation grid already lands inside one xbar's value range — typically the case for low-bit weights / activations or for ideal-baseline experiments. For larger ranges (`Sw > 1` or `Sa > 1`), use [`InterArraySliceXbarMacro`](inter_array_slice.md) or [`IntraArraySliceXbarMacro`](intra_array_slice.md) instead.

## Config sub-modules

`DirectXbarMacroConfig` declares:

- `w_encoding` — signed-digit encoding for the weight transcoder (activations are unsigned true-form by definition).
- `col_accumulator_cfg` — `Tc`-axis cross-tile accumulator.

No `w_slice_num` / `x_slice_num` / shift-adder configs — the direct macro doesn't slice.
