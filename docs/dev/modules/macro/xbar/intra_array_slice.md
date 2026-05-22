# `IntraArraySliceXbarMacro`

Strategy 2: `Sw` gathered within one xbar.

## What one xbar holds

One xbar carries `weights_per_xbar = ⌊col_num / Sw⌋` whole logical weights, with each weight's `Sw` slices in adjacent cols.  Per-xbar effective capacity is `weights_per_xbar × Sw` cells; remaining `col_num - weights_per_xbar × Sw` cells are idle (zero-padded).

`N` logical weights tile into `Tr = ⌈N / weights_per_xbar⌉` row-tiles.

## Organize (W)

`slice` → `[..., N, K, Sw, D]` → pad `N` up to `Tr × weights_per_xbar` → `unflatten(N → Tr, weights_per_xbar)` → tile `K` → permute → `flatten(weights_per_xbar, Sw)` → pad to `col_num` → `Sa=1, M=1` placeholders. Final shape: `[..., M=1, Tc, Tr, Sa=1, data_num=col_num, D, row_num]`.

## Aggregate

Drop trailing idle slots → `unflatten(data_num → weights_per_xbar, Sw)` → `Sw` shift-add (intra-xbar stride-`Sw`) → `Sa` shift-add (intra-xbar serial) → `Tc` accumulate → flatten `(Tr, weights_per_xbar)` → trim to `N`. The macro returns pre-requantize int output; the operator owns bias add and rescale.

## Constraints

- `w_slice_num <= xbar.col_num` (else `weights_per_xbar = 0` would be invalid).

## Config sub-modules

`IntraArraySliceXbarMacroConfig` declares the same fields as `InterArraySliceXbarMacroConfig`.
