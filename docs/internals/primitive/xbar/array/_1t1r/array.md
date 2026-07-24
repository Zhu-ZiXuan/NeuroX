# 1T1R array

`XbarArray1t1r` is the shape-independent pure array: it owns a pluggable [cell](../../cell/_1t1r/cell.md) sub-module, the wire-segment buffers, and the stateless solver. It owns no boundary driver, DAC, reference, or readout — those stay outside as peer blocks, injected into `solve_array` per call.

## Design decisions

- **The array owns a pluggable `cell` sub-module, not the devices directly.** `self.cell = XbarCell1t1r.from_config(config=config.cell_config, policy=policy.cell_policy, inst_shape=..., dtype=..., T__K=...)` resolves the concrete cell model through the 1T1R family root's registry from the `[cim_macro.array_config.cell_config]` sub-tree's `_neurox_class`. Dispatch is family-bounded by type, so the result is always a 1T1R-family cell (`XbarCell1t1r`) with no post-build narrowing — which devices the cell owns, if any, is the cell's own business. The solver and the energy path consume the cell only through its branch / snapshot / `dynamic_energy` contract (see [cell](../../cell/_1t1r/cell.md)), so the array stamps no device current or internal node itself — this keeps it cell-agnostic.
- **Drivers and reference taps are injected, not held.** `solve_array(v_wl, *, bl_driver, bl_v_ref__V, sl_driver, sl_v_ref__V)` takes the two boundary clamp drivers and their pre-snapshotted reference taps as call parameters and constructs no driver, DAC, or reference of its own. This dependency injection is what keeps the array pure.
- **The array takes the analog WL drive, not an integer code.** `solve_array` adds only the size-1 WL-fanout dimension (`v_wl.unsqueeze(-2)`) needed to broadcast the analog drive against the RRAM grid. It performs no code-to-voltage conversion.
- **Energy responsibility split.** `_compute_array_energy__fJ` sums the array-internal **capacitive** energy only — the BL/SL/WL wire-segment cap cycling plus the per-cell node-capacitance energy, delegated to `cell.dynamic_energy(v_bl__V, v_sl__V, solver_dcop.cell, cell_snap)` (passing the converged cell DCOP so the cell charges its own node caps; the array holds no cell-cap formula). The read-current DC conduction ($V \cdot I \cdot t$) is not the array's: it carries a per-input-bit conduction-time weight the axis-agnostic solve cannot apply, so the consuming macro bills it. Peer blocks self-log their own driver / DAC / readout energies.
- **Stream the solve by chunk and release the DCOP each iteration.** The broadcast leading reaches ~$10^5$ instances, so materializing all chunks' node voltages OOMs. Invariant: per-instance memory must stay $O(NB^2)$. Block-Thomas (`solve_block_tridiagonal`) plus per-chunk release holds it, dropping each chunk's `solver_dcop` before the next allocates, so peak working set is $O(\mathrm{chunk}\times\mathrm{col}\times\mathrm{row})$.
- **Compile-path: eager island.** `solve_array` carries `@torch.compiler.disable`. It owns the chunk loop, whose trip count `ceil(leading / solve_chunk_size)` is a runtime value that would unroll and enlarge an enclosing compiled graph if traced. The per-chunk `solver.solve_dc`, itself `@torch.compile(dynamic=False)`, compiles at the fixed `(chunk_size, col, row)` cell-grid shape and is reused across chunks and invocations. See [compile/scheme_a_regional](../../../../compile/scheme_a_regional.md).
- **Wire tensors stay 1-D (no per-instance expansion).** Each BL/SL wire R/G/C buffer is one length-`row` profile shared by all columns; wire R/G feed the solver per call, while wire C feeds the energy model.
- **`wl_pulse_length__ns` is separate from `latency_per_op__ns`.** The pulse length scales the wire-RC charging energy; the per-op latency is attribution only. They are two config fields with disjoint roles — merging them would corrupt the energy.

## Contracts & invariants

- **`solve_array` returns a `XbarArraySteadyState`, not a DCOP.** The frozen struct carries the reassembled `i_bl_port__uA` and `v_bl_clamp__V`, both `[..., num_col]`; energy and latency are emitted internally (`_log_dynamic_energy` + `_log_latency`), not returned. The array builds no DCOP of its own.
- **`weight_grid_shape` property.** The constructor takes the standard `inst_shape` replication prefix plus `row_num` and `col_num`; `weight_grid_shape` derives the full per-cell grid shape `(*self.inst_shape, self._col_num, self._row_num)` as `tuple[int, ...]`. The array owns this concatenation and passes the result to the cell as `inst_shape`; it never derives geometry from cell buffers, which may be 0-d placeholders before `program`.
- **Leading-axis A/B/D classification.** Each broadcast leading position is class A (x-side: `*x_batch`, `M`, `Sa` — independent input vectors, serial in time), class B (inst: `Sw`, `Tc`, `Tr` — independent physical arrays running in parallel), or class D (atomic array: `col`, `row` — one array's internals, **never chunked**). `classify_leading_positions` (`solver/chunking.py`) assigns each from the broadcast's size-1 placeholders. The split drives latency, not chunking: the serial op count is $\prod_{p\in A}\mathrm{leading}[p]$ — B-positions are parallel instances and must not multiply per-op latency. Chunking is axis-agnostic: it slices the flattened A×B leading by `solve_chunk_size`.
- **Snap per chunk.** Each chunk builds one `cell_snap` (`self.cell.snapshot(...)`) plus the per-chunk snaps of the passed-in BL/SL drivers, all at the chunk shape inside the loop; the array re-registers no child state and constructs no driver.
- **Standalone solver access is a supported path.** Tests and solver-calibration tools that need the raw `SolverDcop` construct a solver and call `solver.solve_dc(...)` directly. The array path and direct solver path must stay numerically consistent.

## Performance & resources

Peak working set is $O(\mathrm{chunk}\times\mathrm{col}\times\mathrm{row})$; the single `solve_chunk_size` knob on `XbarArray1t1rPolicy` is the per-chunk leading — the peak-memory budget — trading memory for op count, with `0` selecting the single-block path. At large-transformer-FFN scale an un-chunked leading batch reaches ~$10^5$ positions (~150 GB of node-voltage state), so chunking is mandatory there. The WL drive is materialized once at `(*leading, row)`, not bounded by `solve_chunk_size`. Solver Jacobian storage dominates total memory — see [solver](../../solver.md).

## Gotchas

- **Chunking is bit-exact only under noise-off policies.** Chunking matches the single-block path bit-for-bit only when the policy is deterministic; with noise on, the RNG stream differs across chunk boundaries. Do not assume bit-exactness under noise-on (a concrete scheme's chunking test covers the deterministic case).

---

- **Reference**: [array](../../../../../reference/primitive/xbar/array/_1t1r/array.md)
- **Implementation**: `neurox/primitive/xbar/array/_1t1r.py`
- **Tests**: `tests/primitive/xbar/test_array_fabricate.py`, `tests/works/macro/cim/xue2020jssc/test_energy_accounting.py`
