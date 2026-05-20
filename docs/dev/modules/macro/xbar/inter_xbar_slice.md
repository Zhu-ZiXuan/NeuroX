# `InterXbarSliceMacro`

Strategy 1: `Sw` distributed across xbar planes.

## What one xbar holds

One xbar plane carries one `Sw` slice index across all weights.  For `Sw` slices and `N` logical weights tiled into `Tc` row-tiles, the macro logically uses `Sw × ⌈N / col_num⌉ × ⌈K / row_num⌉` xbar tiles (the current simulator batches them as one tensor for GPU efficiency).

## Organize (W)

Identity. After slicing into `[..., N, K, Sw, D]`, `Sw` stays as a trailing axis through tiling, permute, and placeholder insertion. Final fabricated tensor shape: `[..., M=1, Tc, Tr, Sa=1, Sw, data_num, D, row_num]`.

## Aggregate

`Sa` shift-add (intra-xbar serial) → `Sw` shift-add (cross-xbar weighted-sum) → `Tc` accumulate → flatten `(Tr, data_num)` → trim to `N` → optional bias.

## Config sub-modules

`InterXbarSliceMacroConfig` declares:

- `w_slice_num`, `x_slice_num`, `w_encoding` (activation slicer is always unsigned true-form, no `x_encoding`).
- `col_accumulator_cfg`, `sa_shift_adder_cfg`, `sw_shift_adder_cfg`, `requantizer_cfg`
