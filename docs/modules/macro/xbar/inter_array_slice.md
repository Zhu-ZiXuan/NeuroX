# `InterArraySliceXbarMacro`

Strategy 1: `Sw` distributed across xbar planes.

## What one xbar holds

One xbar plane carries one `Sw` slice index across all weights.  For `Sw` slices and `N` logical weights tiled into `Tc` row-tiles, the macro logically uses `Sw × ⌈N / col_num⌉ × ⌈K / row_num⌉` xbar tiles (the current simulator batches them as one tensor for GPU efficiency).

## Organize (W)

After slicing into `[..., N, K, Sw, D]`, `Sw` is hoisted into the canonical leading position via permute and the `M=1, Sa=1` placeholders are inserted. Final fabricated tensor shape: `[..., M=1, Sa=1, Sw, Tc, Tr, data_num, D, row_num]`. Leading order is `[Sa, Sw, Tc, Tr]` (all four present for InterArray).

## Aggregate

`Sa` shift-add (intra-xbar serial, `dim=-5`) → `Sw` shift-add (cross-xbar weighted-sum, `dim=-4`) → `Tc` accumulate (`dim=-3`) → flatten `(Tr, data_num)` → trim to `N`. The macro returns pre-requantize int output; the operator owns bias add and rescale.

## Config sub-modules

`InterArraySliceXbarMacroConfig` declares:

- `xbar_config: XbarConfig` — owned physical-xbar config.
- `w_slice_num`, `x_slice_num`, `w_encoding` (activation slicer is always unsigned true-form, no `x_encoding`).
- `col_accumulator_config`, `sa_shift_adder_config`, `sw_shift_adder_config`

## Policy

`InterArraySliceXbarMacroPolicy(XbarMacroPolicy)` carries the owned xbar's nonideality policy:

- `xbar: XbarPolicy` — abstract base; the concrete impl is passed by the caller.

The macro forwards `policy.xbar` into `self._build_xbar(...)`.
