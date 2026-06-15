# CircuitCore1T1R — Implementation

## Summary

`CircuitCore1T1R` (`_1t1r/circuit_core.py`) is the shape-independent physical core: it owns the device children (RRAM, access NMOS, TIA, SL driver, WL DAC), the wire-segment buffers, the state-to-conductance lookup, and the stateless solver. It realizes the model in [reference/xbar/_1t1r/circuit_core](../../../reference/xbar/_1t1r/circuit_core.md). This document covers the non-obvious implementation choices, not the read flow.

## Design decisions

- **Stream the solve by chunk and release the DCOP each iteration.** The solve's leading batch is ~$10^5$ (im2col × batch × slice × col); materializing all chunks' node voltages OOMs. Invariant: per-instance memory must stay $O(NB^2)$. Rejected — dense fallback (OOM), PCR with materialized shifts (bloat). Chosen — block-CR plus per-chunk release, dropping each chunk's `solver_dcop` (node voltages, cell currents) before the next allocates, so peak working set is $O(\text{chunk}\times\text{col}\times\text{row})$.
- **Convert the WL DAC before adding the WL-fanout slot.** Converting at full leading without the size-1 fanout slot keeps the DAC's trailing matched to its own `inst_shape`, so numel-based serial accounting, once-per-instance drive noise, and one-event-per-VMM all stay correct. Adding the fanout slot first would multiply the DAC's per-VMM event counts.
- **Wire tensors stay 1-D (no per-instance expansion).** Memory is three length-$N$ tensors per line; wire R/G feed the solver per call, wire C feeds the energy model.
- **`wl_pulse_length__ns` is separate from `latency_per_op__ns`.** The pulse length is a physical duration that multiplies the wire-RC charging integral in the energy model; the per-op latency is attribution only. Merging them would corrupt the energy.

## Contracts & invariants

- **Leading-axis A/B/D classification.** Each broadcast position is class A (x-side: `*x_batch`, `M`, `Sa` — independent input vectors, serialized by `solve_chunk_size_x`), class B (inst: `Sw`, `Tc`, `Tr` — independent physical arrays running in parallel, serialized by `solve_chunk_size_inst`), or class D (atomic core: `col_num`, `row_num` — one array's internals, **never chunked**). `classify_leading_positions` (`_1t1r/_chunking.py`) assigns each from the broadcast's size-1 placeholders. The core's serial op count is $\prod_{p\in A}\text{leading}[p]$ — only A-positions count; B-positions are parallel instances and must not multiply per-op latency.
- **Snapshot per chunk.** Every fabricated device is snapshotted at the chunk shape inside the loop; the core does not re-register child state.
- **The core holds no DCOP of its own.** `cim_read` constructs no `Core1T1RDCOP`; it returns only `v_out_phys`. The downstream consumer needs nothing else.
- **Standalone solver access is a supported path.** Tests and calibration that need the raw `Solver1T1RDCOP` build a solver and call `solver.solve_dc(...)` directly, bypassing `cim_read` (`tests/utils/standalone_solver_fixture.py`, `neurox/tools/solver_calibrate/_common.py`). The two paths must stay numerically consistent.

## Performance & resources

Peak working set is $O(\text{chunk}\times\text{col}\times\text{row})$; `solve_chunk_size_x` (A subset) and `solve_chunk_size_inst` (B subset) on `CircuitCore1T1RPolicy` trade memory for op count, both `0` (single-block) by default. At BERT-FFN scale an un-chunked leading batch reaches ~$10^5$ positions (~150 GB of node-voltage state), so chunking is mandatory there; the per-chunk solver working set is $\text{chunk}_x \cdot \text{chunk}_{\mathrm{inst}} \cdot \text{col} \cdot \text{row}$ per node-voltage field. The WL DAC output is materialized once outside the loop at `(*leading, row)` and is *not* bounded by the chunk knobs — kept un-chunked so each VMM emits exactly one DAC energy + latency event and samples per-instance drive noise once. Solver Jacobian storage dominates total memory — see [solver](solver.md).

## Gotchas

- **Chunking is bit-exact only under noise-off policies.** Either chunk axis matches the single-block path bit-for-bit only when the policy is deterministic; with noise on, the RNG stream differs across chunk boundaries. Do not assume bit-exactness under noise-on (`tests/test_xbar_chunking.py` covers the deterministic case).

## Known limitations

- N/A — no implementation TODOs beyond those tracked in [solver](solver.md).

---

- **Reference**: [circuit_core](../../../reference/xbar/_1t1r/circuit_core.md)
- **Implementation**: `neurox/xbar/_1t1r/circuit_core.py`
- **Tests**: `tests/test_xbar_chunking.py`, `tests/test_xbar_physics.py`
- **Decisions**: N/A — no ADR governs this module.
