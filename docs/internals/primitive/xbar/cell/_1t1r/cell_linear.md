# 1T1R Linear cell

`XbarCell1t1rLinear` (`neurox/primitive/xbar/cell/_1t1r_linear.py`) is the table-driven linearized 1T1R model: no device children, fixed table buffers, ordinary programmed state, and a closed-form branch. The shared 1T1R substrate lives at [1T1R cell base](cell.md).

## Design decisions

- **Program-time gather, gather-free hot path.** `program` validates the state indices (shape = `inst_shape`, range `[0, w_states)`) and gathers the four flat per-state chord-conductance and drop-fraction buffers — each config table indexed directly by state — once into four instance-shaped ordinary attributes (`g_cell_on/off__uS`, `vx_ratio_on/off`). The branch solve then reads the pre-gathered state — the compiled hot path holds no index gather.
- **Pure elementwise branch, division-free, compile-safe.** `solve_branch` is `where` on the WL threshold plus the chord conductance times the terminal drop — no division, no loops, no Python branches on tensor values, no in-place writes. `solve_dc` adds the divider `v_x__V = v_bl - vx_ratio * (v_bl - v_sl)` and, on request, an exactly-zero residual (`zeros_like`): the branch divider satisfies its internal KCL by construction, so there is no convergence knob.
- **Snapshot is expand + chunk-slice, no draws.** `snapshot` broadcasts the four programmed tensors to the per-call shape and selects the chunk positions by `multi_coords` (the same pattern the voltage-driver snapshot uses), bundling them with the WL control in `XbarCell1t1rLinearSnap`. The policy is empty, so the snap is deterministic; `t_elapsed` is unused — the model holds no time-dependent read state.
- **Empty policy, loudly type-checked.** `XbarCell1t1rLinearPolicy` is a truly empty frozen dataclass (every nonideality the model represents is baked into its tables at extraction). `__init__` raises `TypeError` on any other policy type, so wiring a Detail policy TOML against a Linear config fails at construction, not silently.
- **Registry-selected by config type.** The class is decorated `@XbarCell1t1r.register_key(XbarCell1t1rLinearConfig)` on the 1T1R family root; switching a scheme to the Linear model is purely a `_neurox_class` choice on the `cell_config` table, with no code change anywhere.

## Contracts & invariants

- **Signed conductances are exactly $\pm g_{\mathrm{cell}}$.** `solve_branch` returns `(g_cell·ΔV, g_cell, -g_cell)` with $g_{\mathrm{cell}} \ge 0$ guaranteed by config validation (finite, zero allowed — a cut-off branch's honest leakage; array nonsingularity is carried by the wire conductances), so the family sign invariant holds by construction.
- **`program` must precede `snapshot` / solve.** Construction creates only the four small table buffers. `program` creates the four instance-shaped state tensors; an unprogrammed cell has no branch state.

## Performance & resources

The branch solve is a handful of elementwise ops on chunk-shaped tensors — no per-cell iteration, no device model evaluation — so a Linear-cell array solve spends its time in the wire Newton, not the cell. Memory adds four instance-shaped programmed tensors plus the four small table buffers.

## Known limitations

- **Valid near the extraction operating point only.** The tables are chords at one nominal `(v_bl, v_sl)`; the model degrades away from it and represents no per-call stochastic nonideality (see the Reference page's validity section).

---

- **Reference**: [Linear cell](../../../../../reference/primitive/xbar/cell/_1t1r/cell_linear.md)
- **Implementation**: `neurox/primitive/xbar/cell/_1t1r_linear.py`
- **Tests**: `tests/primitive/xbar/test_cell_linear.py`, `tests/tools/test_calibrate_cell_linear.py`
