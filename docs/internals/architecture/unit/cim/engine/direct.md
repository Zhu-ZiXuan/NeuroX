# DirectCimEngine

The no-slice variant. It owns a tile, a weight `Transcoder`, an identity `DirectSlicer` on the activation path, a sub-phase `Accumulator`, and a contraction-tile `Accumulator`, but no shift-adder.

## Design decisions

- **No slice axes in the layout, not size-1 placeholders.** Because `Sw = Sa = 1` are meaningless here, the organized weight tensor `[..., M=1, Tc, Tr, col_num, D, row_num]` simply has no `Sa` / `Sw` axes rather than carrying them as size-1. The `[Sa, Sw, Tc, Tr]` convention applies by omission: the present axes keep their relative order. Only the `M=1` placeholder is inserted, for Cartesian broadcast against the activation.
- **The identity slicer is the semantic witness.** `x_slicer = DirectSlicer(value_range=cim_macro.x_value_range)`: the activation slicing step exists in the pipeline and decomposes nothing — `_organize_x` runs `x_slicer.slice(x).squeeze(-1).squeeze(-1)`, which validates the range elementwise and returns the values unchanged. Tiling is not the slicer's job; it stays in the engine's `_chunk_pad_along` step.
- **Value-range sources.** The ctor sets `_w_value_range` from the transcoder's `value_range` and `_x_value_range` from the `DirectSlicer`'s range (the tile's `x_value_range`); the base publishes them.
- **Construction-time fp32-exactness gate.** `__init__` computes the worst-case per-tile dot `row_num * max|w| * max|x|` from the tile's digit geometry and rejects configurations reaching `2^24`. Below the bound, every per-tile partial survives an fp32 matmul bit-exactly, so the engine stays GPU-capable (CUDA has no integer matmul kernel); the ideal-tile path is elementwise int64 and the physical-tile path is a float solve, both CUDA-safe.

## Contracts & invariants

- **Organized W shape** is `[..., M=1, Tc, Tr, col_num, D, row_num]`; **organized X shape** is `[..., M, Tc, Tr=1, row_num]`. The `Tr=1` placeholder on X lets it broadcast against the W tensor's real `Tr`.
- **Aggregate** is `_unroll_sub_phase → vec_mat_mul → int64 upcast → sub-phase accumulate (dim=_sub_phase_dim = -(b+5), b = len(w_batch)) → Tc accumulate (dim=-3) → flatten (Tr, col_num) → trim to N`. The plane expansion takes `[..., M, Tc, Tr, row_num]` to `[..., P, M, Tc, Tr, row_num]`; the tile read returns per-plane codes `[..., P, *w_batch~, M, Tc, Tr, col_num]` (leading order preserved), and the `phase_accumulator` (inst shape `(w_parallel, Tc, Tr)` — sub-phase-accumulation hardware exists per tile output port) reduces exactly the engine's own P axis before the `Tc` accumulate. There is no shift-add stage. The returned tensor is pre-requantize int.
- **`program` shape gate.** `program(weight)` rejects any shape other than the bound `w_logical_shape`.
- **Out-of-range activations raise.** The `DirectSlicer` range check rejects inputs outside the tile's input grid at `matmul` time; weight ranges remain published, not enforced.

## Performance & resources

The variant adds only the transcode, the identity slice check, and the contraction accumulation; the dominant cost is the tile read. `matmul` runs eager — the heavy DC solve compiles as a separate regional leaf — see [base](base.md).

## Gotchas

- **Last-tile padding is silent.** Trailing columns of the last output tile and trailing rows of the last contraction tile are zero-padded by `_chunk_pad_along`; the padded lanes contribute zero and are trimmed, but a caller inspecting intermediate tile shapes sees the padded extent, not `N` / `K`.

---

- **Reference**: [direct engine](../../../../../reference/architecture/unit/cim/engine/direct.md)
- **Implementation**: `neurox/architecture/unit/cim/engine/direct.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`, `tests/architecture/unit/test_linear_cim_unit.py`
