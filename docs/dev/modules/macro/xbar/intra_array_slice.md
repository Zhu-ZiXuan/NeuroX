# `IntraArraySliceXbarMacro`

Strategy 2: `Sw` gathered within one xbar.

## What one xbar holds

One xbar carries `weights_per_xbar = ⌊col_num / Sw⌋` whole logical weights, with each weight's `Sw` slices in adjacent cols.  Per-xbar effective capacity is `weights_per_xbar × Sw` cells; remaining `col_num - weights_per_xbar × Sw` cells are idle (zero-padded).

`N` logical weights tile into `Tr = ⌈N / weights_per_xbar⌉` row-tiles.

## Organize (W)

`slice` → `[..., N, K, Sw, D]` → pad `N` up to `Tr × weights_per_xbar` → `unflatten(N → Tr, weights_per_xbar)` → tile `K` → permute → `flatten(weights_per_xbar, Sw)` → pad to `col_num` → `M=1, Sa=1` placeholders. Final shape: `[..., M=1, Sa=1, Tc, Tr, data_num=col_num, D, row_num]` — no `Sw` axis (the real `Sw` slices are inlined into `col_num`). Leading order is `[Sa, Sw, Tc, Tr]`; `Sw` is absent.

## Aggregate

Drop trailing idle slots → `unflatten(data_num → weights_per_xbar, Sw)` → `Sw` shift-add (intra-xbar stride-`Sw`, `dim=-1`) → `Sa` shift-add (intra-xbar serial, `dim=-4`) → `Tc` accumulate (`dim=-3`) → flatten `(Tr, weights_per_xbar)` → trim to `N`. The macro returns pre-requantize int output; the operator owns bias add and rescale.

## Constraints

- `w_slice_num <= xbar.col_num` (else `weights_per_xbar = 0` would be invalid).

## Config sub-modules

`IntraArraySliceXbarMacroConfig` declares the same fields as `InterArraySliceXbarMacroConfig`:

- `xbar_config: XbarConfig` — owned physical-xbar config.
- `w_slice_num`, `x_slice_num`, `w_encoding`.
- `col_accumulator_config`, `sa_shift_adder_config`, `sw_shift_adder_config`.

## Policy

`IntraArraySliceXbarMacroPolicy(XbarMacroPolicy)` carries the owned xbar's nonideality policy:

- `xbar: XbarPolicy` — abstract base; the concrete impl is passed by the caller.

The macro forwards `policy.xbar` into `self._build_xbar(...)`.
