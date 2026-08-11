# Xbar cell base

`XbarCell` defines a condensed two-terminal branch together with the `XbarCellConfig`, `XbarCellPolicy`, `XbarCellSnap`, and `XbarCellDcop` bases. It owns device children and per-call electrical state but does not report PPA events itself.

## Design decisions

- **The cell remains in the module tree without reporting PPA.** Device children require module registration so buffers follow `.to(...)` / `.eval()` and fabrication reaches them. `is_profile_target = False` excludes the cell from static collection and forbids direct dynamic-event emission; the owning array bills the cell's nodes and logs on its behalf. `XbarCellConfig` therefore carries no shared PPA fields.
- **The base is a pure contract.** `XbarCell` defines neither a registry nor a factory. A family that needs config-keyed dispatch owns that mechanism at its own root, keeping the base independent of every concrete family.
- **Empty container bases, shared DCOP fields.** `XbarCellConfig`, `XbarCellPolicy`, and `XbarCellSnap` are marker bases because no config, policy, or snap field is universal. `XbarCellDcop` owns the common condensed branch triple (`i__uA`, `di_dvbl__uS`, `di_dvsl__uS`); subclasses may add internal state. A model-specific residual is a separate probe payload, never a DCOP field.
- **Type-parameterised so the typed surface survives subclassing.** `XbarCell` is `Generic[ConfigT, PolicyT, SnapT, DCOPT]`; the call pair `SnapT` / `DCOPT` (bound to `XbarCellSnap` / `XbarCellDcop`) is what the typed call surface rides. A concrete cell binds the pair to its own subclasses, so `snapshot` / `solve_branch` / `solve_dc` carry the concrete types with no LSP-narrowing override — these typed surfaces stay type-checked across topologies.
- **Internal nodes condensed in the cell, not the array solve.** `solve_branch` eliminates every internal node inside the cell and exposes only the condensed two-terminal branch through the typed `XbarCell` surface. The cell therefore binds to no particular solver and carries no solver state — it is a stateless per-call actor.
- **The cell answers for its branch and accounts for nothing.** A cell config carries no capacitance at all: the node-capacitance inventory and the whole account sit on the owning array, which is what holds the rails, the rest boundary an access departs from, and the grid each node's share of line follows. The array's only data dependency on the cell is the typed call outputs — the DCOP and the snap the cell already returns — plus the instance properties the cell declares; no owner reads a field back out of the child config it handed over. The law those bills follow is [capacitive energy](../../../../reference/primitive/physics.md).
- **A model-specific residual uses a colocated probe link.** When a cell exposes a residual diagnostic, its concrete module defines the payload, prober, and emission frequency. Cells without such a diagnostic define no probe link.

## Contracts & invariants

- **Abstract method set.** Every implementation supplies `snapshot`, `program`, `solve_branch`, and `solve_dc`. `solve_branch` returns only `(i__uA, di_dvbl__uS, di_dvsl__uS)`; `solve_dc` returns the corresponding `DCOPT` superset. Their branch triples must be identical for the same inputs and snap.
- **Signed-conductance contract.** Every implementation returns `di_dvbl__uS` non-negative and `di_dvsl__uS` non-positive. The base does not validate either sign.
- **Control and device samples travel in the snap.** `snapshot(control=..., shape=..., t_elapsed=...)` bundles the device snaps with the cell's exogenous control drive; `solve_branch` / `solve_dc` read the control only from the snap, never from `self`, so a single `snap` carries all per-call control and the cell holds no per-call control state. `control` arrives per cell at `[..., col, row]` — it is each cell's own access-device gate voltage, and whichever lines the owning array shares between cells is the array's business to express when it builds the tensor. `shape` is the per-call broadcast shape `(..., col, row)`, which the cell forwards to its device snaps so device samples stay aligned with the broadcast operating point. There is no chunk-selection argument: a snap is sampled once at the full call shape, and any memory tiling slices the snap object afterwards.
- **`@torch.compile` constraints.** `solve_branch` executes inside a compiled region, so every concrete implementation obeys the same rules: no in-place tensor writes, no Python-side branches on tensor values.

---

- **Reference**: [cell family](../../../../reference/primitive/xbar/cell/family.md)
- **Implementation**: `neurox/primitive/xbar/cell/base.py`
- **Tests**: TODO — no dedicated base-contract test
