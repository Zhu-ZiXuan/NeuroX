# `neurox/xbar/solver.py`

## Current role

`solver.py` holds the **shared** circuit-solver math utilities reused across crossbar IR-drop solvers. It owns no topology and no solver framework — every helper takes its tensors as plain arguments.

The solver framework (base class + registry + DCOP / Residuals containers) lives with its topology family. For 1T1R, see `docs/modules/xbar/_1t1r/solver.md`.

## Exposed helpers

- `elementwise_diff(fn, *, wrt, **kwargs)` — per-element derivative of an elementwise device function via autograd. Because the function is elementwise, `∂(Σ y) / ∂x[k] = ∂y[k] / ∂x[k]`, so one backward pass yields the full per-element Jacobian diagonal without ever materialising off-diagonal terms.

- `solve_tridiagonal(sub, diag, sup, rhs, dim)` — batched Thomas-algorithm solver for scalar tridiagonal systems. Forward elimination accumulates the modified diagonal and RHS slices into Python lists (no in-place writes), then back-substitution builds the solution right-to-left and stacks. List-based accumulation avoids tensor mutation, which is incompatible with `torch.compile` graph tracing; Inductor then fuses the per-step elementwise ops across the unrolled loop.

- `solve_block_tridiagonal(sub, diag, sup, rhs)` — block-Thomas algorithm for block-tridiagonal systems with `B × B` blocks. Used by `NestedSolver1T1R`'s coupled BL/SL wire Newton (`B = 2`) and `FullJacobianSolver1T1R`'s monolithic Newton step (`B = 3`). Shape convention is fixed: block tensors are `[..., N, B, B]`, rhs is `[..., N, B]`. The `B = 1` case reduces to scalar Thomas (matches `solve_tridiagonal` numerically up to fp64 round-off). Each `B × B` block-inverse step uses `torch.linalg.solve`.

- `col_wire_kcl_residual(...)` / `col_driver_current(...)` — KCL residual and boundary current builders for per-column wire ladders. Used by both nested and full-Jacobian solvers; shared so the residual diagnostics stay consistent across solvers.

## Why these live in `solver.py`

The helpers are crossbar-topology-agnostic numerical primitives. Keeping them here separates them from any one topology's Newton loop.

## `@torch.compile` friendliness

All helpers run as part of the compiled DC solve. The implementation constraints reflect this:

- no in-place tensor writes
- no Python-side scalar branches on tensor values
- no autograd reads of attributes that change across calls

These constraints are documented inside the helpers themselves; respecting them yields a single fused kernel covering the whole solve.

See also:

- `docs/modules/xbar/_1t1r/solver.md` — 1T1R solver framework (base + registry + DCOP)
- `docs/modules/xbar/_1t1r/nested_solver.md` — `NestedSolver1T1R`
- `docs/modules/xbar/_1t1r/full_jacobian_solver.md` — `FullJacobianSolver1T1R`
- `docs/modules/tools/solver_calibrate/README.md` — calibration methodology
- `docs/modules/analog/tia/base.md` — BL boundary actor (TIA family)
- `docs/modules/analog/driver.md` — SL boundary actor
