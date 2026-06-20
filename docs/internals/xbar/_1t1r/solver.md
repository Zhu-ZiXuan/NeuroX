# DC Solver — Implementation

## Summary

The 1T1R DC solve is realized as a `RegistryMixin` framework (`Solver1T1R` base + `Solver1T1RDCOP` / `Solver1T1RResiduals` containers in `_1t1r/solver.py`), shared linear-algebra helpers (`xbar/solver.py`), and the nested concrete solver (`nested_solver.py`). The numerical method, well-posedness, and the cell-pluggable contract are specified in [reference/xbar/_1t1r/solver](../../../reference/xbar/_1t1r/solver.md); this document covers the implementation choices and trade-offs.

## Design decisions

- **Solvers are stateless tool classes, not `nn.Module`s.** They carry no buffers and no registered state, so they stay out of the core's module tree — otherwise the fabricate cascade and `state_dict` would wrongly sweep them in. The cell holds the device state; the solver holds the method (see [cell](cell.md)).
- **The solver consumes a pluggable cell, never a device.** `Solver1T1R.__init__` binds three boundary actors — the cell, the BL driver (TIA), and the SL driver. The solver calls `cell.solve_branch(v_bl, v_sl, cell_snap)` at each wire-Newton step and assembles the block-$2\times2$ wire Jacobian from the returned single current and two signed conductances; it stamps no device current and models no internal cell node. This is what lets one solver serve any cell satisfying the contract and is the core reason the solver carries no global block-tridiagonal formulation — see [ADR-0003](../../../about/adr/ADR-0003-xbar-cell-abstraction-and-single-nested-solver.md).
- **Per-family base, not a generic `Solver`.** A generic base needed `**kwargs` for the boundary handles, which defeats type-checking; the 1T1R base spells its three boundary actors (`cell`, `bl_driver`, `sl_driver`) in the signature. Other topology families define their own base rather than subclassing this.
- **Solvers have no Policy.** Every knob is either a workload-tuned numerical constant (`Solver1T1RConfig` subclass) or a method-intrinsic safety bound (`ClassVar` on the solver), neither of which is a per-source non-ideality toggle. So `CircuitCore1T1RPolicy` has no `solver` field.
- **Decompose the solve (nested) rather than one simultaneous Newton over all unknowns.** A simultaneous Newton over every wire / clamp unknown admits rail pseudo-equilibria and can leave a uA-scale BL wire residual; the nested block-Gauss-Seidel split into well-posed sub-problems eliminates both — see the [reference well-posedness argument](../../../reference/xbar/_1t1r/solver.md#well-posedness).
- **Iteration counts come from `solver_calibrate`, not hand-tuning** (chip-parameter-free step-ratio plateau; see [calibration guide](../../../guides/calibration/README.md)). The picker emits the raw plateau plus a fixed $+1$ outer margin; the resulting `n_outer` / `n_inner` are stored in the chip config (`[xbar.core_config.solver_config]`), which is their only authoritative home. The per-cell condensation count `n_newton` is calibrated separately and owned by the cell (see [cell](cell.md)).
- **Helpers accumulate into lists, never in-place.** In-place tensor writes break `torch.compile` graph tracing; list accumulation lets inductor fuse the unrolled loop.
- **Method-intrinsic damping caps are `ClassVar`s, not config.** They are per-iteration $|\Delta u|$ damping bounds (a Newton heuristic, not a chip parameter), so they never enter the calibrated config or any Policy. The nested solver carries an outer cap $\texttt{MAX\_OUTER\_STEP\_\_V} = 0.10$ (applied to the coupled $2\times2$ clamp step) and an inner cap $\texttt{MAX\_INNER\_STEP\_\_V} = 0.05$ (applied to the block-$2\times2$ wire step).

### Nested solver structure

The nested solver splits the coupled system into two well-posed sub-problems and iterates them block-Gauss-Seidel style. A warm start seeds the IR-drop wire voltages and the boundary clamps; a coupled outer Newton then steps the clamp-driver pair $(V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}})$ (clamp-first, $2\times2$ per column, carrying the BL $\leftrightarrow$ SL cross-coupling so a variable SL driver needs no special case), and for each outer step an inner array solve runs a coupled block-$2\times2$ wire Newton for $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ at the frozen clamps. Each inner step calls `cell.solve_branch` for the per-node branch current and signed conductances — the cell condenses its own internal node, so the wire Newton sees a two-terminal element, not a device stack. Both Newtons are damped by per-iteration $|\Delta u|$ caps. A final cell refresh (`cell.solve_dc`) aligns the returned cell DCOP and internal node with the returned wire voltages.

The solve has two kinds of scalar: calibrated iteration counts (`NestedSolver1T1RConfig`, loaded from the chip config) and method-intrinsic damping caps (`ClassVar`s baked into the solver):

| Knob | Kind | Role | Value |
|---|---|---|---|
| `n_outer` | config count | coupled outer Newton steps on $(V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}})$ | `[xbar.core_config.solver_config].n_outer` |
| `n_inner` | config count | inner block-Newton steps on the wire system per outer step | `[xbar.core_config.solver_config].n_inner` |
| `MAX_OUTER_STEP__V` | ClassVar cap | outer-step $\lvert\Delta u\rvert$ damping cap | 0.10 |
| `MAX_INNER_STEP__V` | ClassVar cap | inner-step $\lvert\Delta u\rvert$ damping cap | 0.05 |

`n_outer` / `n_inner` are the margined calibration plateau (raw plateau plus the picker's $+1$ outer margin) and live in the chip config — see the design-decision bullet above and the [calibration guide](../../../guides/calibration/README.md). The damping caps are Newton heuristics fixed in the solver source, not chip parameters.

## Contracts & invariants

- **The cell is the only device interface.** The solver reads device behaviour exclusively through `cell.solve_branch` (hot path) and `cell.solve_dc` (final refresh), passing the `cell_snap` it received. It depends on the cell honouring its contract: one branch current per site, $\partial I/\partial V_{\mathrm{BL}} \ge 0$, $\partial I/\partial V_{\mathrm{SL}} \le 0$. A cell that violated the signs would silently break the wire M-matrix rather than error.
- **`@torch.compile` constraints (whole solve).** No in-place tensor writes; no Python-side branches on tensor values; no autograd reads of attributes that change across calls. Respecting them yields a single fused kernel over the solve.
- **Shared containers decouple the caller from the solver.** `Solver1T1RDCOP` carries the wire / clamp state plus the condensed cell DCOP (`Solver1T1RDCOP.cell`); `Solver1T1RResiduals` carries the wire / clamp KCL residuals, while the per-cell internal-KCL residual lives on the cell DCOP. The core's data interface does not change with the solver.
- **Residuals are opt-in.** They are computed only when `compute_residuals=True`; the hot path leaves them `None` so the extra evaluations are elided from the compiled graph.

## Performance & resources

mid1 preset, inst=64, batch=64, torch.compile, median of 5:

| dtype | latency [ms] | peak mem [GB] |
|---|---|---|
| fp32 | 582 | 2.7 |
| fp64 | 3050 | 5.4 |

fp32 is the chip-preset default (~5× faster at ~50% memory); fp64 is for accuracy verification. Condensing the cell's internal node removes one unknown per cell from the array solve, so the wire Jacobian stays block-$2\times2$ (`[..., col, row, 2, 2]`) rather than carrying the internal node — the memory that lets fp64 / large-batch workloads run. Chunking (see [circuit_core](circuit_core.md)) is the handle when other GPU users compete.

## Gotchas

- **`B=1` block-Thomas is not bit-identical to scalar Thomas** — see [xbar/solver](../solver.md). The wire Newton uses $B=2$, but the same primitive's $B=1$ path differs from scalar division by fp64 round-off.

## Known limitations

- **Cell condensation is verified through residuals, not a global Jacobian.** With the single nested formulation, correctness rests on the converged-residual checks (wire / clamp residuals on the solver DCOP, per-cell internal-KCL residual on the cell DCOP) and the physical-vs-ideal agreement — not on cross-agreement with a second formulation. Those checks are the standing evidence; see [ADR-0003](../../../about/adr/ADR-0003-xbar-cell-abstraction-and-single-nested-solver.md).

---

- **Reference**: [solver](../../../reference/xbar/_1t1r/solver.md)
- **Implementation**: `neurox/xbar/solver.py`, `neurox/xbar/_1t1r/solver.py`, `neurox/xbar/_1t1r/nested_solver.py`
- **Tests**: `tests/test_xbar_physics.py`, `tests/test_xbar_chunking.py`
- **Decisions**: [ADR-0003 pluggable xbar cell and the single nested solver](../../../about/adr/ADR-0003-xbar-cell-abstraction-and-single-nested-solver.md)
