# DirectXbarMacro — Implementation

## Summary

`DirectXbarMacro` (`xbar/direct.py`): the no-slice mode. It owns a tile, a weight `Transcoder`, and a contraction-tile `Accumulator`, but no slicer and no shift-adder. Spec: [reference/macro/xbar/direct](../../../reference/macro/xbar/direct.md).

## Design decisions

- **No slice axes in the layout, not size-1 placeholders.** Because `Sw = Sa = 1` are meaningless here, the organized weight tensor `[..., M=1, Tc, Tr, col_num, D, row_num]` simply has no `Sa` / `Sw` axes rather than carrying them as size-1. The `[Sa, Sw, Tc, Tr]` convention applies by omission: the present axes keep their relative order. Only the `M=1` placeholder is inserted, for Cartesian broadcast against the activation.
- **Value-range delegation.** `w_value_range` returns the transcoder's `value_range`; `x_value_range` returns the tile's `x_range`. The mode holds no range of its own and enforces nothing — out-of-range inputs propagate.

## Contracts & invariants

- **Organized W shape** is `[..., M=1, Tc, Tr, col_num, D, row_num]`; **organized X shape** is `[..., M, Tc, Tr=1, row_num]`. The `Tr=1` placeholder on X lets it broadcast against the W tensor's real `Tr`.
- **Aggregate** is `vec_mat_mul → Tc accumulate (dim=-3) → flatten (Tr, col_num) → trim to N`. There is no shift-add stage. The returned tensor is pre-requantize int; bias and rescale are the operator's.
- **`program` shape gate.** `program(weight)` rejects any shape other than the bound `w_logical_shape`.

## Performance & resources

The mode adds only the transcode and the contraction accumulation; the dominant cost is the tile read. `matmul` runs eager — the heavy DC solve compiles as a separate regional leaf — see [base](base.md).

## Gotchas

- **Last-tile padding is silent.** Trailing columns of the last output tile and trailing rows of the last contraction tile are zero-padded by `chunk_pad_along`; the padded lanes contribute zero and are trimmed, but a caller inspecting intermediate tile shapes sees the padded extent, not `N` / `K`.

## Known limitations

- N/A.

---

- **Reference**: [direct](../../../reference/macro/xbar/direct.md)
- **Implementation**: `neurox/macro/xbar/direct.py`
- **Tests**: `tests/test_xbar_macro.py`
- **Decisions**: N/A — no ADR governs this module.
