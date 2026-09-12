# Adaptive solve validation

Validation checks the [parallel BL/SL equations](../reference/primitive/xbar/solver/col_bl_col_sl.md), their numerical solution, and integration into array and macro execution. Electrical tests independently verify that convergence criteria identify the intended equilibrium.

## Electrical checks

The focused suite in `tests/primitive/xbar/test_col_bl_col_sl.py` uses linear cell branches with ideal and resistive clamps to compare against an independently assembled dense nodal system. It checks the single-row closed form, reconstructs terminal wire and port residuals from the returned electrical state, and exercises both supported dtypes across wire resistances. These checks distinguish a correct equilibrium from a permissive stopping threshold.

The same suite checks independent column termination, bounded voltage updates, immediate rejection of non-finite branch states, strict rejection when a node solve reaches its cap, and non-strict return of the requested failure trace. Shape cases include one row, one column, and cell-controlled leading axes broadcast over singleton clamp snapshots.

Detailed-cell tests reconstruct the internal KCL and verify both capped behaviors: the ordinary solve raises, while the traced solve returns its unconverged terminal sample. Linear-cell extraction uses converged DCOPs.

## Observation and integration checks

Solver tests compare traced and untraced electrical results for convergent workloads and check the residual observations against each concrete trace contract. Array tests compare electrical results across chunk sizes with the same sampled physical state. Macro integration tests check output codes and profiling effects.

The focused CPU regression command is:

```bash
uv run pytest tests/primitive/xbar tests/works/macro/cim/test_solver_diagnostics.py --device cpu
```

## Compiled execution checks

The solver suite captures a full graph with snapshot-aliasing cell derivatives and checks that changing iteration caps leaves recursive graph-node counts unchanged. This checks bounded graph structure at a fixed geometry.

Run the solver suite on a selected CUDA device to exercise compiled numerical loops:

```bash
uv run pytest tests/primitive/xbar/test_col_bl_col_sl.py --device cuda:0
```

A performance measurement fixes the physical inputs, geometry, dtype, device, trace request, chunk size, and software environment. Measure cold compilation with independent caches, warm execution with repeated synchronized calls, and peak device allocation separately. Retain workload definitions and environment metadata with the run outputs so measurements can be repeated.
