# xbar cell base — Implementation

## Summary

The pluggable-cell family: the abstract `XbarCell` (`xbar/cell.py`) plus its `XbarCellConfig` / `XbarCellPolicy` / `XbarCellSnap` / `XbarCellDCOP` / `XbarCellResiduals` container bases. A concrete cell is documented under [the concrete cell page](_1t1r/cell.md). Spec: [reference/xbar/cell](../../reference/xbar/cell.md).

## Design decisions

- **The cell is a `FabricateMixin` + `nn.Module`; it is not a `CircuitBase`.** A cell owns its device children, so it must be in the module tree: `nn.Module` so buffers move with `.to` / `.eval`, and `FabricateMixin` so the manufacturing-variation cascade reaches the devices. It deliberately is **not** a `CircuitBase` — it carries no PPA. The owned devices' area / leakage roll up through the owning core's budget; the cell contributes only per-read device-capacitor switching energy via `dynamic_energy`. That is why `XbarCellConfig` is a no-PPA base.
- **Config-keyed registry dispatch, no `isinstance` ladder.** `XbarCell` is `RegistryMixin[type[XbarCellConfig], XbarCell]`: each concrete cell registers against the `XbarCellConfig` subclass it consumes (`@XbarCell.register_key(SomeCellConfig)`), and `from_config(config=...)` dispatches on `type(config)`. Adding a cell adds a registration line and never touches `from_config`. Same owned-construction discipline as the rest of NeuroX (see [ADR-0001](../../about/adr/ADR-0001-config-dispatch-and-owned-construction.md)); the consuming core builds its cell from its `cell_config`.
- **Empty container bases, not shared fields.** `XbarCellConfig` / `XbarCellPolicy` / `XbarCellSnap` / `XbarCellResiduals` are markers: the family has no field every topology shares, so each concrete cell carries its own subclass (device configs, composite per-device policy, device snaps plus control state, internal residuals). Only `XbarCellDCOP` is a real base — the condensed branch triple $(I, \partial I/\partial V_{\mathrm{BL}}, \partial I/\partial V_{\mathrm{SL}})$ plus an optional `residuals` is the solver-facing surface every cell returns identically, so it is concrete and shared, and concrete cells subclass it to add their internal-node voltages.
- **Type-parameterised so the typed surface survives subclassing.** `XbarCell` is `Generic[SnapT, DCOPT]` (bound to `XbarCellSnap` / `XbarCellDCOP`), and `XbarCellDCOP` is in turn `Generic[ResidualsT]` (bound to `XbarCellResiduals`). A concrete cell binds all three vars to its own subclasses, so `snapshot` / `solve_branch` / `solve_dc` / `dynamic_energy` and the DCOP `residuals` field carry the concrete types with no LSP-narrowing override — the solver- and calibration-facing surfaces stay type-checked across topologies.
- **Internal nodes condensed in the cell, not the solver.** The cell eliminates every internal node inside `solve_branch`, so the array solver consumes a two-terminal branch and stays cell-agnostic. This is why the array solver carries no global block-tridiagonal formulation — see [ADR-0003](../../about/adr/ADR-0003-xbar-cell-abstraction-and-single-nested-solver.md). The array solver is generic, not topology-bound: it consumes this condensed branch through the typed `XbarCell[SnapT, DCOPT]` surface, taking the cell as one of its swappable per-call actors alongside the clamp drivers — the cell is not fixed to any one solver.

## Contracts & invariants

- **Family init signature.** `__init__(*, config, policy, inst_shape, dtype, T__K)` — all keyword-only. The base stores `self.config` and `self._inst_shape` and discards `policy / dtype / T__K` (consumed by the concrete init); `FabricateMixin` provides the auto-cascade `fabricate()` and concrete cells fan the mismatch sample into their device children rather than holding cell-level mismatch.
- **The abstract method set every cell implements.** `snapshot`, `program`, `solve_branch`, `solve_dc`, `dynamic_energy` are `@abstractmethod`; `from_config` is the only concrete entry point. `solve_branch` is the lean hot path returning `(i__uA, di_dvbl__uS, di_dvsl__uS)`; `solve_dc` is its diagnostic / energy superset returning a concrete `XbarCellDCOP` with internal-node state and, when `compute_residuals=True`, the internal-KCL residual. Both must condense identically.
- **Signed-conductance contract.** `di_dvbl__uS` is non-negative and `di_dvsl__uS` is non-positive at every operating point. The array wire Jacobian assembled by the solver depends on these signs; a cell that violated them would silently corrupt the wire solve rather than raise.
- **Control and device samples travel in the snap.** `snapshot(control=..., shape=..., multi_coords=..., t_elapsed=...)` bundles the device snaps with the cell's exogenous control drive; `solve_branch` / `solve_dc` / `dynamic_energy` read the control only from the snap, never from `self`, so the solver passes one `cell_snap` and the cell holds no per-call control state. `shape` is the per-call broadcast shape `(*leading, col, row)` and `multi_coords` the advanced-index tuple selecting one chunk's positions (`None` for the full view); the cell forwards both to its device snaps so device samples stay aligned with the broadcast operating point.
- **`@torch.compile` constraints.** `solve_branch` runs inside the consuming compiled solver leaf, so every concrete implementation obeys the same rules: no in-place tensor writes, no Python-side branches on tensor values.

## Performance & resources

N/A at this level — the memory- and compile-sensitive work (the condensation, its iteration budget, per-chunk working set) is topology-specific; see [the concrete cell page](_1t1r/cell.md).

## Gotchas

- **Do not treat the cell as a `CircuitBase`.** It has no `area_per_inst__um2` / `leakage_per_inst__uW`; querying PPA on the cell is a category error. PPA lives on the owning core; the cell owns only dynamic device-cap energy.
- **`dynamic_energy` needs a converged DCOP, not just terminal voltages.** It reads the internal-node voltages off the concrete `XbarCellDCOP` to charge the internal device caps; pass the DCOP from `solve_dc`, not the terminal voltages alone.

## Known limitations

- N/A at the family level — see [the concrete cell page](_1t1r/cell.md) for the concrete cell's limitations.

---

- **Reference**: [cell](../../reference/xbar/cell.md)
- **Implementation**: `neurox/xbar/cell.py`
- **Tests**: `tests/test_xbar_cell.py`
- **Decisions**: [ADR-0004 clamp-driver role and the topology-agnostic array solver](../../about/adr/ADR-0004-clamp-driver-protocol-and-generic-solver.md), [ADR-0003 pluggable xbar cell and the single nested solver](../../about/adr/ADR-0003-xbar-cell-abstraction-and-single-nested-solver.md)
