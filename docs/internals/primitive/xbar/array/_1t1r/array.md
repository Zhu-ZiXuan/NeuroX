# 1T1R array

`XbarArray1T1R` is the shape-independent pure array: it owns a pluggable [cell](../../cell/_1t1r/cell.md) sub-module, the wire-segment buffers, and the stateless solver. It owns no boundary driver, DAC, reference, or readout — those stay outside as peer blocks, injected into `solve_array` per call.

## Design decisions

- **The array owns a pluggable `cell` sub-module, not the RRAM / NMOS directly.** `self.cell = XbarCell.from_config(config=config.cell_config, ...)` constructs the cell from the `[cim_macro.array_config.cell_config]` sub-tree; the RRAM and access NMOS are the cell's own children (`self.cell.rram` / `self.cell.nmos`). The solver and the energy path consume the cell only through its branch / snapshot / `dynamic_energy` contract (see [cell](../../cell/_1t1r/cell.md)), so the array stamps no device current or internal node itself — this keeps it cell-agnostic.
- **Drivers and reference taps are injected, not held.** `solve_array(v_wl, *, bl_driver, bl_v_ref__V, sl_driver, sl_v_ref__V)` takes the two boundary clamp drivers and their pre-snapshotted reference taps as call parameters and constructs no driver, DAC, or reference of its own. This dependency injection is what keeps the array pure.
- **The array takes the analog WL drive, not the integer code.** `solve_array` receives the analog WL drive and only adds the size-1 WL-fanout slot (`v_wl.unsqueeze(-2)`) it needs to broadcast against the RRAM grid; it never touches an integer activation code. The DAC conversion and its drive-noise / per-VMM accounting sit upstream.
- **Energy responsibility split.** `_compute_array_energy__fJ` sums the array-internal energy and delegates the per-cell device-capacitance energy to `cell.dynamic_energy(v_bl__V, v_sl__V, solver_dcop.cell, cell_snap)`, passing the converged cell DCOP so the cell charges its own internal-node caps; the array holds no device-cap formula. Peer blocks self-log their own driver / DAC / readout energies.
- **Stream the solve by chunk and release the DCOP each iteration.** The leading batch reaches ~$10^5$ (im2col × batch × slice × line), so materializing all chunks' node voltages OOMs. Invariant: per-instance memory must stay $O(NB^2)$. Rejected — dense fallback (OOM), PCR with materialized shifts (bloat). Chosen — block-Thomas (`solve_block_tridiagonal`) plus per-chunk release, dropping each chunk's `solver_dcop` before the next allocates, so peak working set is $O(\mathrm{chunk}\times\mathrm{line}\times\mathrm{series})$.
- **Compile-path: eager island.** `solve_array` carries `@torch.compiler.disable`. It owns the chunk loop, whose trip count `ceil(leading / solve_chunk_size)` is a runtime value that would unroll and explode an enclosing compiled graph if traced. Keeping it eager pins the loop in Python while the per-chunk `solver.solve_dc` — itself `@torch.compile(dynamic=False)` — compiles once at the fixed chunk shape (chunk indexing flattens every call to one `(chunk_size, 1, series)`) and is reused across every chunk, VMM, and caller instance. See [compile/scheme_a_regional](../../../../compile/scheme_a_regional.md).
- **Wire tensors stay 1-D (no per-instance expansion).** Memory is three length-$N$ tensors per line; wire R/G feed the solver per call, wire C feeds the energy model.
- **`wl_pulse_length__ns` is separate from `latency_per_op__ns`.** The pulse length scales the wire-RC charging energy; the per-op latency is attribution only. They are two config fields with disjoint roles — merging them would corrupt the energy.

## Contracts & invariants

- **`solve_array` returns a `XbarArraySteadyState`, not a DCOP.** The frozen struct carries the reassembled `i_bl_port__uA` and `v_bl_clamp__V`, both `[..., num_line]`; energy and latency are emitted internally (`_log_dynamic_energy` + `_log_latency`), not returned. The array builds no DCOP of its own; further fields are added only if a test needs them.
- **`weight_grid_shape` property.** Exposes `self.cell.rram.g__uS.shape` as `tuple[int, ...]` so a caller can size a full-leading per-instance input.
- **Leading-axis A/B/D classification.** Each broadcast leading position is class A (x-side: `*x_batch`, `M`, `Sa` — independent input vectors, serial in time), class B (inst: `Sw`, `Tc`, `Tr` — independent physical arrays running in parallel), or class D (atomic array: `line`, `series` — one array's internals, **never chunked**). `classify_leading_positions` (`solver/chunking.py`) assigns each from the broadcast's size-1 placeholders. The split drives latency, not chunking: the serial op count is $\prod_{p\in A}\mathrm{leading}[p]$ — B-positions are parallel instances and must not multiply per-op latency. Chunking is axis-agnostic: it slices the flattened A×B leading by `solve_chunk_size`.
- **Snap per chunk.** Each chunk builds one `cell_snap` (`self.cell.snapshot(...)`) plus the per-chunk snaps of the passed-in BL/SL drivers, all at the chunk shape inside the loop; the array re-registers no child state and constructs no driver.
- **Standalone solver access is a supported path.** Tests and calibration that need the raw `SolverDCOP` build a solver and call `solver.solve_dc(...)` directly, bypassing `solve_array` (`tests/utils/standalone_solver_fixture.py`, the chip solver calibrator). The two paths must stay numerically consistent.

## Performance & resources

Peak working set is $O(\mathrm{chunk}\times\mathrm{line}\times\mathrm{series})$; the single `solve_chunk_size` knob on `XbarArray1T1RPolicy` is the per-chunk leading — the peak-memory budget — trading memory for op count, with `0` selecting the single-block path. At large-transformer-FFN scale an un-chunked leading batch reaches ~$10^5$ positions (~150 GB of node-voltage state), so chunking is mandatory there. The WL drive is materialized once at `(*leading, series)`, not bounded by `solve_chunk_size`. Solver Jacobian storage dominates total memory — see [solver](../../solver.md).

## Gotchas

- **Chunking is bit-exact only under noise-off policies.** Chunking matches the single-block path bit-for-bit only when the policy is deterministic; with noise on, the RNG stream differs across chunk boundaries. Do not assume bit-exactness under noise-on (a concrete scheme's chunking test covers the deterministic case).

---

- **Reference**: [array](../../../../../reference/primitive/xbar/array/_1t1r/array.md)
- **Implementation**: `neurox/primitive/xbar/array/_1t1r.py`
- **Tests**: a concrete scheme's xbar chunking and physics tests exercise this array.
