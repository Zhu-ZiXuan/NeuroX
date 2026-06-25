# 1T1R Cell — Implementation

## Summary

The pluggable-cell abstraction is a `RegistryMixin` family (`XbarCell` base + `XbarCellConfig` / `XbarCellPolicy` / `XbarCellSnap` / `XbarCellDCOP` / `XbarCellResiduals` containers in `xbar/cell.py`) with one 1T1R implementation (`xbar/cell_1t1r.py`). The cell realizes the model in [reference/xbar/cell_1t1r](../../../reference/xbar/_1t1r/cell.md). This document covers the non-obvious choices, not the branch-solve flow.

## Design decisions

- **The cell is a `FabricateMixin` + `nn.Module`, the solver is not.** A cell owns its RRAM / NMOS device children, so it must be in the module tree: `nn.Module` so buffers move with `.to` / `.eval` and `FabricateMixin` so the manufacturing-variation cascade reaches the devices. The array solver, by contrast, is a stateless tool class kept out of the module tree (see [solver](../solver.md)); it is generic and consumes this cell as one of its swappable per-call actors, alongside the clamp drivers, rather than being bound to it. The cell holds the state; the solver holds the method.
- **Config-keyed registry dispatch, no universal `from_config`.** `XbarCell` is `RegistryMixin[type[XbarCellConfig], XbarCell]`: each concrete cell registers against the `XbarCellConfig` subclass it consumes (`@XbarCell.register_key(XbarCell1T1RConfig)`), and `XbarCell.from_config(config=...)` dispatches on `type(config)`. Same owned-construction discipline as the rest of NeuroX (see [ADR-0001](../../../about/adr/ADR-0001-config-dispatch-and-owned-construction.md)); the core constructs its cell from `config.cell_config`.
- **Internal node condensed in the cell, not the solver.** $V_{\mathrm{X}}$ is eliminated by a per-cell Newton inside `solve_branch`, so the solver consumes a two-terminal branch and stays cell-agnostic. This is why the solver carries no global block-tridiagonal formulation — see [ADR-0003](../../../about/adr/ADR-0003-xbar-cell-abstraction-and-single-nested-solver.md).
- **Pade current-divider seed before the Newton loop.** Seeding $V_{\mathrm{X}}$ from a first-order conductance-divider split of the BL-to-SL drop lands the iterate inside the Newton basin, so a small fixed step count converges. A cold seed would need more steps or risk a bad first step on the stiff NMOS / RRAM I-V.
- **Fixed unrolled `n_newton`, no convergence branch.** The Newton loop runs exactly `n_newton` steps with no data-dependent stopping test. A runtime `while` on a tensor residual breaks `torch.compile` tracing; a fixed trip count keeps the cell body inside the one compiled solver leaf. The count is calibrated, not guessed (next bullet).
- **`n_newton` is calibrated per cell type and owned by the cell config.** It is a numerical-convergence knob, not a chip-physics parameter and not a per-source nonideality toggle, so it lives on `XbarCell1T1RConfig` (calibrated by step-ratio plateau, see [calibration guide](../../../guides/calibration/README.md)) and never on a Policy. Each cell type calibrates its own count because the condensation it solves is its own.
- **No PPA on the cell.** `XbarCell` is not a `CircuitBase`: its config carries no area / leakage fields. The owned devices' physical area / leakage roll up through the core's PPA budget; the cell contributes only per-VMM device-capacitor switching energy via `dynamic_energy`.

## Contracts & invariants

- **`solve_branch` is the lean hot path; `solve_dc` is its diagnostic superset.** `solve_branch` returns only `(i__uA, di_dvbl__uS, di_dvsl__uS)` — exactly what the wire Newton needs. `solve_dc` returns the same plus the internal-node voltage on the concrete DCOP and, when `compute_residuals=True`, the internal-KCL residual. The hot path calls `solve_branch`; energy and calibration call `solve_dc`. Both must condense identically.
- **Signed-conductance contract.** `di_dvbl__uS` is non-negative and `di_dvsl__uS` is non-positive at every operating point, by series condensation of the device partials (RRAM and NMOS-drain partials $\ge 0$, NMOS-source partial $\le 0$). The wire Jacobian assembled by the solver depends on these signs; a cell that violated them would silently corrupt the wire solve rather than error.
- **Single current.** The branch current returned is the RRAM current `i_r`; at cell convergence it equals the NMOS current. `solve_dc(compute_residuals=True)` exposes their absolute difference as the per-cell KCL residual — the calibration / debug signal for whether `n_newton` is sufficient.
- **Control line travels in the snap.** `snapshot(control=..., shape=..., multi_coords=..., t_elapsed=...)` bundles the RRAM / NMOS device snaps with the WL drive (`v_wl__V`); `solve_branch` / `solve_dc` / `dynamic_energy` read the WL only from the snap, never from `self`. So the solver passes one `cell_snap` and the cell holds no per-call control state.
- **`snapshot` threads the broadcast shape and chunk selection to its devices.** `shape` is the per-call broadcast shape `(*leading, col, row)` and `multi_coords` is the advanced-index tuple selecting one chunk's positions (`None` for the full view); the cell forwards both straight to `rram.snapshot` / `nmos.snapshot`. This keeps the device samples (`g__uS`, `vth__V`, `beta__uA_per_V2`) at the same broadcast leading as the WL / wire tensors the solver drives, so chunked / batched reads stay shape-aligned.
- **`program` writes only the storage device.** It maps a state-index tensor through `state_to_g_map__uS` and programs the RRAM; the NMOS is not programmed. The state-index shape must match the cell's `inst_shape`.
- **`@torch.compile` constraints.** `solve_branch` runs inside the compiled solver leaf, so it obeys the same rules: no in-place tensor writes, no Python-side branches on tensor values. The unrolled Newton and the functional (non-in-place) updates satisfy this.

## Performance & resources

The cell's per-call working set is the device snaps plus a handful of node-voltage-shaped tensors at the per-call (chunked) leading; it allocates no $V_{\mathrm{X}}$ history across Newton steps (each step overwrites the iterate functionally). `n_newton` is small (single digits at fp32), so the unrolled loop adds a constant multiple of one RRAM + one NMOS `solve_dc` per cell per solver iteration. The condensation removes one unknown per cell from the array solve entirely — there is no $V_{\mathrm{X}}$ in the solver's wire Jacobian, which is the memory win that lets the wire Newton stay block-$2\times2$ (see [solver](../solver.md)).

## Gotchas

- **Do not treat the cell as a `CircuitBase`.** It has no `area_per_inst__um2` / `leakage_per_inst__uW`; querying PPA on the cell is a category error. PPA lives on the owning core; the cell owns only dynamic device-cap energy.
- **`dynamic_energy` needs a converged DCOP, not just terminal voltages.** It reads $V_{\mathrm{X}}$ from the `XbarCell1T1RDCOP` to charge the bottom-electrode and drain-body caps; calling it with terminal voltages alone would miss the internal-node caps. Pass the DCOP from `solve_dc`.
- **`n_newton` is per cell type, not shared with the solver counts.** It is distinct from the solver's `n_outer` / `n_inner` and from the TIA's `n_newton`; recalibrating one does not recalibrate the others.

## Known limitations

- **Cell-internal convergence is verified through residuals, not a closed-form root.** The per-cell KCL residual (`solve_dc(compute_residuals=True)`) is the standing check that the fixed `n_newton` condenses $V_{\mathrm{X}}$ to the numerical floor; there is no separate analytic-root cross-check. The device-derivative signs are checked directly — the test asserts the returned conductances are non-negative / non-positive (footer's Tests).

---

- **Reference**: [cell](../../../reference/xbar/_1t1r/cell.md)
- **Implementation**: `neurox/xbar/cell.py`, `neurox/xbar/cell_1t1r.py`
- **Tests**: `tests/test_xbar_cell.py`, `tests/test_xbar_physics.py`
- **Decisions**: [ADR-0004 clamp-driver role and the topology-agnostic array solver](../../../about/adr/ADR-0004-clamp-driver-protocol-and-generic-solver.md), [ADR-0003 pluggable xbar cell and the single nested solver](../../../about/adr/ADR-0003-xbar-cell-abstraction-and-single-nested-solver.md)
