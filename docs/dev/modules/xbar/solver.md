# `neurox/xbar/solver.py`

## Current role

`solver.py` holds the **shared** circuit-solver utilities reused across crossbar IR-drop solvers. It owns no topology — every helper takes its tensors as plain arguments.

## Exposed helpers

- `elementwise_diff(fn, *, wrt, **kwargs)` — per-element derivative of an elementwise device function via autograd. Because the function is elementwise, `∂(Σ y) / ∂x[k] = ∂y[k] / ∂x[k]`, so one backward pass yields the full per-element Jacobian diagonal without ever materialising off-diagonal terms.

- `solve_tridiagonal(sub, diag, sup, rhs, dim)` — batched Thomas-algorithm solver. Forward elimination accumulates the modified diagonal and RHS slices into Python lists (no in-place writes), then back-substitution builds the solution right-to-left and stacks. List-based accumulation avoids tensor mutation, which is incompatible with `torch.compile` graph tracing; Inductor then fuses the per-step elementwise ops across the unrolled loop.

## Why these live in `solver.py`

The helpers are crossbar-topology-agnostic numerical primitives. Keeping them here separates them from any one topology's Newton loop.

## `@torch.compile` friendliness

Both helpers run as part of the compiled DC solve. The implementation constraints reflect this:

- no in-place tensor writes
- no Python-side scalar branches on tensor values
- no autograd reads of attributes that change across calls

These constraints are documented inside the helpers themselves; respecting them yields a single fused kernel covering the whole solve.

See also:

- `_1t1r/newton_raphson_solver.md`
- `../analog/clamp_driver.md`
