# Xbar DC solver base

`Solver` is the interface of one stateless DC solve over a fabricated tile, together with the shared `SolverConfig` and `SolverDcop` containers. It fixes the call surface a solve presents to its owning array — the cell, the two boundary clamps, and the rail scalars all arrive per call — and owns no numerical method itself.

## Design decisions

- **A solver is stateless and fixed to one array topology, while remaining independent of concrete cell and driver implementations.** `Solver.__init__` takes only its numerical config and binds no boundary actors. `solve_dc` receives the cell, the BL driver, and the SL driver (plus their snaps) per call; an implementation stamps no device current, models no internal cell node, and hard-wires no concrete driver class.
- **The class is non-generic; `solve_dc` is method-generic.** The cell and driver types vary per call, not per construction, so the type parameters live on the method. `solve_dc` is generic over `(CellSnapT, CellDCOPT, BLSnapT, SLSnapT)`: the cell types retain their `XbarCellSnap` / `XbarCellDcop` bounds, and the driver snaps retain their `ClampSnap` bound. `SolverDcop` is `Generic[CellDCOPT]`, so the return type preserves the concrete cell DCOP instead of widening it to `Any`. Method-level parameters avoid a `**kwargs` boundary signature while keeping the complete surface type-checked.
- **Solvers are stateless tool classes, not `nn.Module`s.** They carry no buffers and no registered state, so they stay out of the core's module tree — otherwise the fabricate cascade and `state_dict` would wrongly sweep them in. The cell holds the device state; the solver holds the method (see [cell](../cell/base.md)).
- **The owning array selects its solver explicitly.** A solver is a numerical implementation of one array topology, not a configurable physical module family. The owning array therefore constructs it directly from the matching `SolverConfig` subclass; neither the solver nor its config participates in registry dispatch. The boundary actors remain per-call arguments of `solve_dc`, not construction-time fields.
- **The cell-grid layout is a fixed solver contract.** Cell-grid tensors use `[..., col, row]`, matching the array layout. The last axis follows the wire ladder through the rows, while the second-to-last axis indexes independent columns and their BL/SL driver pairs. The solver neither accepts an axis selector nor recursively rewrites cell snaps and DCOPs. A consuming array with another internal layout must organize its solver-boundary data into this contract because that array, not the numerical solver, owns the physical meaning of its dimensions.
- **Solvers have no Policy.** Every knob is either a workload-tuned numerical constant (`SolverConfig` subclass) or a method-intrinsic safety bound (`ClassVar` on the solver); none is a per-source non-ideality toggle.

## Contracts & invariants

- **The cell is the only device interface; the drivers are the only boundary interface.** A solve reads device behaviour exclusively through the cell's condensed-branch surface (`cell.solve_branch`, `cell.solve_dc`) and boundary behaviour through each driver's clamp transfer, passing the snaps it received. It consumes the cell's condensed-branch and signed-conductance contract unchecked, so a cell that violates the signs silently corrupts the solve rather than raising.
- **`SolverDcop` is the complete solve result.** It carries the wire and clamp state plus the condensed cell DCOP (`SolverDcop.cell`) behind one solver-independent data contract.
- **The rail lattice reaches the solve as two scalars.** `solve_dc` declares `bl_segment_r__MOhm` and `sl_segment_r__MOhm` as plain Python floats and derives the link conductance `1 / r` at entry, uS being exactly the reciprocal of MOhm. Nothing below that entry holds a per-row profile, so a rail whose links differ node by node is outside this interface rather than a value it can carry.
- **Degenerate extents are not guarded.** A single row, a single column, and a single block row are all well-defined systems the same code path settles, so nothing rejects them. Only checks that prevent silent corruption survive anywhere in this package — the chunking layer's leading-shape check and each algorithm's own shape assumptions; a one-position axis is not a corruption and a `> 1` precondition would only have blocked the smallest useful test case.

---

- **Reference**: N/A — the base is a software contract; each solver's formulation is specified in its own Reference document
- **Implementation**: `neurox/primitive/xbar/solver/base.py`
- **Tests**: TODO — no dedicated base-contract test
