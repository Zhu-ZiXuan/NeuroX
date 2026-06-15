# DC Solvers — Implementation

## Summary

The 1T1R DC solve is realized as a `RegistryMixin` framework (`Solver1T1R` base + `Solver1T1RDCOP` / `Solver1T1RResiduals` containers in `_1t1r/solver.py`), shared linear-algebra helpers (`xbar/solver.py`), and two concrete solvers (`nested_solver.py`, `full_jacobian_solver.py`). The numerical method, well-posedness, and convergence are specified in [reference/xbar/_1t1r/solver](../../../reference/xbar/_1t1r/solver.md); this document covers the implementation choices and trade-offs.

## Design decisions

- **Solvers are stateless tool classes, not `nn.Module`s.** They carry no buffers and no registered state, so they stay out of the core's module tree — otherwise the fabricate cascade and `state_dict` would wrongly sweep them in.
- **Per-family base, not a generic `Solver`.** A generic base needed `**kwargs` for device handles, which defeats type-checking; the 1T1R base spells its four boundary actors (`rram`, `nmos`, `bl_driver`, `sl_driver`) in the signature. Other topology families define their own base rather than subclassing this.
- **Solvers have no Policy.** Every knob is either a workload-tuned numerical constant (`Solver1T1RConfig` subclass) or a method-intrinsic safety bound (`ClassVar` on the solver), neither of which is a per-source non-ideality toggle. So `CircuitCore1T1RPolicy` has no `solver` field.
- **Decompose the solve (nested) rather than one simultaneous Newton over all unknowns.** A simultaneous Newton admits rail pseudo-equilibria and can leave a uA-scale BL wire residual; the nested block-Gauss-Seidel split into well-posed sub-problems eliminates both — see the [reference well-posedness argument](../../../reference/xbar/_1t1r/solver.md#well-posedness).
- **Nested is the production default; full-Jacobian is debug / cross-check only.** Nested wins on memory in both dtypes and scales to larger batches; full-Jacobian exists so the two can be cross-validated against each other (cross-check test in the footer's Tests).
- **Iteration counts come from `solver_calibrate`, not hand-tuning** (chip-parameter-free step-ratio plateau; see [calibration guide](../../../guides/calibration/README.md)). At fp32 the plateau lands at nested $(n_\text{outer}, n_\text{inner}) = (3, 1)$ and full-Jacobian $n = 3$; the preset adds a $+1$ margin, so the $(4, 1)$ / $4$ counts in the table below are margined, not raw, values.
- **Helpers accumulate into lists, never in-place.** In-place tensor writes break `torch.compile` graph tracing; list accumulation lets inductor fuse the unrolled loop.
- **Method-intrinsic damping caps are `ClassVar`s, not config.** They are per-iteration $|\Delta u|$ damping bounds (a Newton heuristic, not a chip parameter), so they never enter the calibrated config or any Policy. Nested carries an outer cap $\texttt{MAX\_OUTER\_STEP\_\_V} = 0.10$ (applied to the coupled $2\times2$ clamp step) and an inner cap $\texttt{MAX\_INNER\_STEP\_\_V} = 0.05$ (applied to the block-$2\times2$ wire step); full-Jacobian carries a single $\texttt{MAX\_STEP\_\_V} = 0.05$ applied component-wise across all five unknown classes of $u$.

### Nested solver structure

The nested solver splits the coupled system into two well-posed sub-problems and iterates them block-Gauss-Seidel style. A warm start seeds $V_X$, the IR-drop wire voltages, and the boundary clamps; a coupled outer Newton then steps the clamp/drive pair $(V_\text{BL,CL}, V_\text{SL,DR})$ (clamp-first, $2\times2$ per column, carrying the BL $\leftrightarrow$ SL cross-coupling so a variable SL driver needs no special case), and for each outer step an inner array block-Newton resolves the per-row $(V_\text{BL}, V_\text{SL}, V_X)$ wire system at the frozen clamps. Both Newtons are damped by per-iteration $|\Delta u|$ caps. A final cell refresh aligns the returned cell currents and $V_X$ with the returned wire voltages.

The iteration knobs (`NestedSolver1T1RConfig`; full-Jacobian's `n_newton` for parallel) and the method-intrinsic damping caps (`ClassVar`s) at the fp32 chip preset are

| Knob | Role | Value |
|---|---|---|
| `n_outer` | coupled outer Newton steps on $(V_\text{BL,CL}, V_\text{SL,DR})$ | 4 |
| `n_inner` | inner block-Newton steps on the wire system per outer step | 1 |
| `MAX_OUTER_STEP__V` | outer-step $\lvert\Delta u\rvert$ damping cap | 0.10 |
| `MAX_INNER_STEP__V` | inner-step $\lvert\Delta u\rvert$ damping cap | 0.05 |
| `MAX_STEP__V` | full-Jacobian single-step $\lvert\Delta u\rvert$ damping cap | 0.05 |

The `n_outer`, `n_inner` counts are the margined calibration plateau (raw $(3, 1)$ plus the preset's $+1$ outer margin); see the design-decision bullet above and the [calibration guide](../../../guides/calibration/README.md). Full-Jacobian uses `n_newton = 4` (margined from a raw plateau of 3).

### Full-Jacobian assembly

The full-Jacobian solver lifts every circuit unknown into one global Newton step. Per `(batch, col)` instance the unknown vector is $u = [V_\text{BL}[0{:}R],\, V_\text{SL}[0{:}R],\, V_X[0{:}R],\, V_\text{BL,CL},\, V_\text{SL,DR}]$ with $R$ the row count. This single-step assembly is the **analytic baseline that the FD-verify test cross-checks** (`test_full_jacobian_fd_verify.py`): the closed-form stamping below is what the finite-difference Jacobian's block-tridiagonal *structure* is asserted against.

The per-instance Jacobian is **block-tridiagonal with $3\times3$ diagonal blocks**: each diagonal block couples the $(V_\text{BL}, V_\text{SL}, V_X)$ triple of one wire row through the local RRAM/NMOS conductances, the off-diagonal blocks carry the BL-BL and SL-SL wire couplings between adjacent rows, and the two boundary scalars $V_\text{BL,CL}$, $V_\text{SL,DR}$ are Schur-eliminated against row 0 before the block solve. The reduced system is solved with `solve_block_tridiagonal(block_size=3)`.

Linearising the clamp residual $F_\text{CL,BL}$ gives the closed form

$$\Delta v_\text{clamp} = \alpha + \beta \cdot \Delta v_\text{bl}[0]$$

with

$$\beta = \frac{-\,r_\text{driver}\, g_\text{seg}[0]}{1 - r_\text{driver}\, g_\text{seg}[0]}, \qquad \alpha = \frac{-\,F_\text{CL,BL}(u)}{1 - r_\text{driver}\, g_\text{seg}[0]}.$$

The wire-row-0 diagonal entry for $V_\text{BL}$ absorbs $-g_\text{seg}[0] \cdot \beta$, and the right-hand side at the same slot picks up $+g_\text{seg}[0] \cdot \alpha$; the recovered $\Delta v_\text{clamp}$ is added back to $V_\text{BL,CL}$ after the block solve. The SL boundary uses the identical construction.

## Contracts & invariants

- **`@torch.compile` constraints (whole solve).** No in-place tensor writes; no Python-side branches on tensor values; no autograd reads of attributes that change across calls. Respecting them yields a single fused kernel over the solve.
- **Shared containers decouple the caller from the solver.** `Solver1T1RDCOP` / `Solver1T1RResiduals` are shared by both solvers, so switching solver does not change the core's data interface.
- **Residuals are opt-in.** They are computed only when `compute_residuals=True`; the hot path leaves them `None` so the extra evaluations are elided from the compiled graph.

## Performance & resources

mid1 preset, inst=64, batch=64, torch.compile, median of 5:

| dtype | solver | latency [ms] | peak mem [GB] |
|---|---|---|---|
| fp32 | nested (4,1) | 582 | **2.7** |
| fp32 | full-Jacobian (4) | **319** | 3.8 |
| fp64 | nested (5,3) | 3050 | 5.4 |
| fp64 | full-Jacobian (5) | 1586 | 7.5 |

fp32 is ~5× faster at ~50% memory → the chip-preset default; fp64 is for accuracy verification. Full-Jacobian's dense block-$3\times3$ Jacobian is `[..., col, row, 3, 3]` (~1.9 GB fp64 on mid1); nested's block-$2\times2$ wire Jacobian uses ~4× less and scales where full-Jacobian OOMs at fp64. Chunking (see [circuit_core](circuit_core.md)) is the handle when other GPU users compete.

## Gotchas

- **Full-Jacobian's lower fp32 latency does not make it a production path.** Its role is solver-correctness debug, FD-Jacobian verification, and small-workload cross-check. Treating it as a production fallback OOMs at fp64 / large batch. For a one-off cross-check, point `[xbar.core_config.solver_config]._neurox_type` at `FullJacobianSolver1T1RConfig`; do not leave it there.

## Known limitations

- **No element-wise analytical-vs-FD Jacobian comparison.** `tests/test_full_jacobian_fd_verify.py` checks the converged $u^\*$ satisfies an independent residual, that the FD Jacobian is non-singular, and that its block-tridiagonal *structure* matches the closed form — but not that the analytical stamping matches FD entry-by-entry. Closing this gap needs the internal Jacobian-assembly path exposed to tests; deferred until symbolic-derivation bugs are suspected.

---

- **Reference**: [solver](../../../reference/xbar/_1t1r/solver.md)
- **Implementation**: `neurox/xbar/solver.py`, `neurox/xbar/_1t1r/solver.py`, `neurox/xbar/_1t1r/nested_solver.py`, `neurox/xbar/_1t1r/full_jacobian_solver.py`
- **Tests**: `tests/test_full_jacobian_fd_verify.py`, `tests/test_full_jacobian_solver.py`, `tests/test_xbar_chunking.py`
- **Decisions**: N/A — no ADR governs this module.
