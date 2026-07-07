# 1T1R core array

## Summary

`Core1T1R` (`xbar/_1t1r/core.py`) is the shape-independent pure array: it owns a pluggable [cell](cell.md) sub-module (which owns the RRAM + access NMOS), the wire-segment buffers, and the stateless solver. It does **not** own drivers, a DAC, a reference source, or a readout — those are peers hoisted above it on the scheme xbar. This document covers the non-obvious implementation choices, not the read flow.

## Design decisions

- **The core owns a pluggable `cell` sub-module, not the RRAM / NMOS directly.** `self.cell = XbarCell.from_config(config=config.cell_config, ...)` constructs the cell from the `[xbar.core_config.cell_config]` sub-tree; the RRAM and access NMOS are the cell's children, reached via `self.cell.rram` / `self.cell.nmos`. The solver and the energy path consume the cell through its branch / snapshot / `dynamic_energy` contract (see [cell](cell.md)), so the core never stamps a device current or an internal node itself — that is what keeps the core cell-agnostic.
- **The core takes drivers in, it does not hold them.** `solve_array(v_wl, *, bl_driver, bl_v_ref__V, sl_driver, sl_v_ref__V)` receives the analog WL drive (the xbar already ran its WL DAC) and the two boundary clamp drivers plus their pre-snapshotted reference taps from the scheme xbar. The core snapshots the passed-in drivers per chunk and threads the snaps into `solve_dc`; it constructs no driver, DAC, or reference of its own. This is what demotes the core to a pure array.
- **Energy split: device caps in the cell, everything else array-internal in the core.** `_compute_array_energy__fJ` sums the array-internal terms — DC conduction at the clamps, BL/SL wire-segment caps, and the WL-line cap — and delegates the per-cell device-capacitance switching energy to `cell.dynamic_energy(v_bl, v_sl, solver_dcop.cell, cell_snap)`, summed over the array. The core passes the converged cell DCOP (carrying the condensed internal node) so the cell can charge its internal-node caps; the core itself holds no device-cap formula. The driver / DAC / readout energies belong to the scheme xbar's peers, which self-log them.
- **Stream the solve by chunk and release the DCOP each iteration.** The solve's leading batch is ~$10^5$ (im2col × batch × slice × line); materializing all chunks' node voltages OOMs. Invariant: per-instance memory must stay $O(NB^2)$. Rejected — dense fallback (OOM), PCR with materialized shifts (bloat). Chosen — block-Thomas (`solve_block_tridiagonal`) plus per-chunk release, dropping each chunk's `solver_dcop` (node voltages, cell currents) before the next allocates, so peak working set is $O(\mathrm{chunk}\times\mathrm{line}\times\mathrm{series})$.
- **Compile-path: no (eager island).** `solve_array` carries `@torch.compiler.disable`. It owns the chunk loop, whose trip count is a runtime value that would unroll and explode an enclosing compiled graph if traced. Keeping it eager pins the loop in Python while the per-chunk `solver.solve_dc` — itself `@torch.compile(dynamic=False)` — compiles once at the fixed chunk shape (the chunk indexing flattens every call to one `(chunk_size, 1, series)`) and is reused across every chunk, VMM, and caller instance. See [compile/scheme_a_regional](../../compile/scheme_a_regional.md).
- **The core takes the analog WL drive, not the integer code.** `solve_array` receives the analog WL drive at full leading and adds only the size-1 fanout slot it needs for the broadcast. The DAC conversion (and its once-per-instance drive noise / one-event-per-VMM accounting) lives upstream on a `wl_dac` peer; the core never touches the integer activation code.
- **Wire tensors stay 1-D (no per-instance expansion).** Memory is three length-$N$ tensors per line; wire R/G feed the solver per call, wire C feeds the energy model.
- **`wl_pulse_length__ns` is separate from `latency_per_op__ns`.** The pulse length is a physical duration that multiplies the wire-RC charging integral in the energy model; the per-op latency is attribution only. Merging them would corrupt the energy.

## Contracts & invariants

- **`solve_array` returns a `CoreSteadyState`.** The frozen struct carries the reassembled fields the readout needs — `i_bl_port__uA` and `v_bl_clamp__V`, both `[..., num_line]`. Energy is emitted internally (`_log_dynamic_energy` + `_log_latency`), not in the struct; further DCOP fields are added only if a test needs them.
- **`weight_grid_shape` property.** The core exposes `weight_grid_shape -> tuple[int, ...]` (the `self.cell.rram.g__uS.shape`) so a caller can size its per-instance input to full leading before its own drive conversion — preserving the per-instance drive-noise behavior.
- **Leading-axis A/B/D classification.** Each broadcast position is class A (x-side: `*x_batch`, `M`, `Sa` — independent input vectors, serial in time), class B (inst: `Sw`, `Tc`, `Tr` — independent physical arrays running in parallel), or class D (atomic core: `line`, `series` — one array's internals, **never chunked**). `classify_leading_positions` (`solver/chunking.py`) assigns each from the broadcast's size-1 placeholders. The classification is used for latency, not chunking: the core's serial op count is $\prod_{p\in A}\mathrm{leading}[p]$ — only A-positions count; B-positions are parallel instances and must not multiply per-op latency. Chunking itself is axis-agnostic (it slices the flattened A×B leading by `solve_chunk_size`).
- **Snap per chunk.** Each chunk builds one `cell_snap` (`self.cell.snapshot(control=v_wl_chunk, ...)`, which bundles the RRAM / NMOS device snaps with the WL drive) plus the per-chunk snaps of the **passed-in** BL/SL drivers, all at the chunk shape inside the loop; the core does not re-register child state and constructs no driver.
- **The core holds no DCOP of its own.** `solve_array` constructs no `Core1T1RDCOP`; it returns a `CoreSteadyState`. The downstream readout needs nothing else.
- **Standalone solver access is a supported path.** Tests and calibration that need the raw `SolverDCOP` build a solver and call `solver.solve_dc(...)` directly, bypassing `solve_array` (`tests/utils/standalone_solver_fixture.py`, the chip solver calibrator). The two paths must stay numerically consistent.

## Performance & resources

Peak working set is $O(\mathrm{chunk}\times\mathrm{line}\times\mathrm{series})$; the single `solve_chunk_size` knob on `Core1T1RPolicy` is the per-chunk leading — the peak-memory budget — trading memory for op count, `0` (single-block) by default. At large-transformer-FFN scale an un-chunked leading batch reaches ~$10^5$ positions (~150 GB of node-voltage state), so chunking is mandatory there; the per-chunk solver working set is $\mathrm{solve\_chunk\_size} \cdot \mathrm{line} \cdot \mathrm{series}$ per node-voltage field. The WL drive handed in by the xbar is materialized once at `(*leading, series)` and is *not* bounded by `solve_chunk_size`. Solver Jacobian storage dominates total memory — see [solver](../solver.md).

## Gotchas

- **Chunking is bit-exact only under noise-off policies.** Chunking matches the single-block path bit-for-bit only when the policy is deterministic; with noise on, the RNG stream differs across chunk boundaries. Do not assume bit-exactness under noise-on (a concrete scheme's chunking test covers the deterministic case).

## Known limitations

- N/A — no implementation TODOs beyond those tracked in [solver](../solver.md).

---

- **Reference**: [core](../../../reference/xbar/_1t1r/core.md)
- **Implementation**: `neurox/xbar/_1t1r/core.py`
- **Tests**: a concrete scheme's xbar chunking and physics tests exercise this core.
