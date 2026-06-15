# `neurox/xbar/_1t1r/full_jacobian_solver.py`

## Current role

`FullJacobianSolver1T1R` is the fully-coupled-Jacobian DC solver for a fabricated 1T1R tile. Every circuit unknown — `V_BL[k]`, `V_SL[k]`, `V_X[k]`, plus the per-column `V_BL_CL` and `V_SL_DR` — enters one global Newton step. Per-iteration it assembles the block-tridiagonal Jacobian, Schur-eliminates the two boundary scalars against row 0 of the array, solves the reduced system with `solve_block_tridiagonal(block_size=3)`, and recovers the eliminated `ΔV_clamp` from the recovered row-0 update.

Compared with `NestedSolver1T1R`'s block-Gauss-Seidel decomposition this captures every coupling simultaneously — no per-axis decomposition, no nested fixed point on `V_clamp`. The trade is one larger linear solve per Newton iteration (block-3×3 Thomas) against fewer outer Newton iterations and no TIA Newton iterations multiplied by an outer-loop iteration count.

## Unknown vector + residuals

Per `(batch, col)` instance, the unknown vector is

```
u = [V_BL[0:R], V_SL[0:R], V_X[0:R], V_BL_CL, V_SL_DR]
```

with `R = num_row`. The five residual classes are

```
F_X[k]  = I_N(V_WL[k], V_X[k], V_SL[k]) − I_R(V_BL[k] − V_X[k])
F_BL[k] = wire_BL[k](V_BL, V_BL_CL) + I_R(V_BL[k] − V_X[k])
F_SL[k] = wire_SL[k](V_SL, V_SL_DR) − I_N(V_WL[k], V_X[k], V_SL[k])
F_CL_BL = V_BL_CL − TIA(I_BL_PORT)
F_CL_SL = V_SL_DR − SL_driver(I_SL_PORT)
```

with `I_BL_PORT = g_seg[0]·(V_BL_CL − V_BL[0])` and analogously for SL. `F_BL` uses `I_R` (RRAM current, the BL-side injection), `F_SL` uses `I_N` (NMOS current, the SL-side draw); at convergence the two agree.

## Jacobian assembly

For each wire row `k` the diagonal 3×3 block (ordering `[V_BL, V_SL, V_X]`) is

```
[ bl_wire_diag[k] + g_R[k]   0                          −g_R[k]            ]
[ 0                          sl_wire_diag[k] − g_NS[k]  −g_ND[k]           ]
[ −g_R[k]                    g_NS[k]                    g_ND[k] + g_R[k]   ]
```

where `g_R, g_ND, g_NS` are the public local derivatives from `RRAM.solve_dc` and `NMOS.solve_dc`. Off-diagonal blocks carry only the BL-BL and SL-SL wire couplings (`−g_seg`) — the rows for `V_X` have no inter-row coupling, so columns 2 of `sub`/`sup` are all zero.

The two boundary residuals are Schur-eliminated before the block solve: linearising `F_CL_BL` gives `Δv_clamp = α + β·Δv_bl[0]` with

```
β = −r_driver · g_seg[0] / (1 − r_driver · g_seg[0])
α = −F_CL_BL(u) / (1 − r_driver · g_seg[0])
```

The wire-row-0 diagonal entry for `V_BL` absorbs `−g_seg[0]·β`; the rhs at the same slot picks up `+g_seg[0]·α`. The recovered `Δv_clamp` is added to `V_BL_CL` after the block solve. Same construction for SL.

## Class constants

| Attribute | Value | Purpose |
|---|---|---|
| `MAX_STEP__V` | 0.05 V | Per-iteration `|Δu|` damping cap; applied component-wise to all five unknown classes. Newton heuristic, not chip-tuneable. |

Anything you want to chip-tune lives in `FullJacobianSolver1T1RConfig` instead.

## Calibration

`solver_calibrate.full_jacobian` performs a single-axis sweep over `n_newton` using the same chip-parameter-free framework as `solver_calibrate.nested` (see that file's "Calibration methodology" section):

1. **Primary**: step-ratio plateau on the global unknown vector `u = [V_BL, V_SL, V_X, V_BL_CL, V_SL_DR]`.
2. **Sanity guard**: relative residual check against workload-derived signal scales.

Quadratic Newton convergence means most chips reach round-off within a few iterations. Calibrated at **fp32** (production dtype) on the mid1 chip preset, the step plateau lands at `n_newton = 3`; with margin `+1` the recommended value is `n_newton = 4`. At fp64 the plateau lands at `n_newton = 4` (recommended 5 with margin) — one more step buys 9 more orders of magnitude in the residual but is overkill for chip simulation.

## Verification

`tests/test_full_jacobian_fd_verify.py` builds a tiny standalone harness (single column, 4 rows) and:

1. Asserts the solver-converged `u*` satisfies an INDEPENDENT residual function `F(u*) ≈ 0` (catches: wrong fixed point).
2. Computes the FD Jacobian `J_FD = ∂F/∂u` at `u*` and verifies the dense Newton step `−J_FD⁻¹·F(u*)` is small. Since `F(u*) ≈ 0` from step 1, this would be tautologically small for ANY invertible operator — the assertion only verifies `J_FD` is non-singular and the FD harness itself is healthy. It does **not** prove the solver's analytical Jacobian matches FD entry-wise.
3. Checks the FD Jacobian's block-tridiagonal structure matches the doc's closed-form recipe (V_X has no inter-row coupling, BL-BL/SL-SL wire couplings have the right sign and magnitude, boundary rows touch only V_BL[0]/V_SL[0] and V_clamp/V_drive respectively). This is the strongest current signal for the analytical stamping being correct.

A true element-wise analytical-vs-FD Jacobian comparison would require exposing the solver's internal Jacobian-assembly path to tests. Not done today — flagged for future work if symbolic-derivation bugs are suspected.

## Memory budget

Per VMM the dense block-3×3 Jacobian carries `[..., col, row, 3, 3]` entries — for the mid1 chip's `row = 64, col = 68, batch ≤ 2048` that's `~1.9 GB` at fp64 (`~0.95 GB` at fp32). Chunking via `CircuitCore1T1RPolicy.solve_chunk_size_x` / `solve_chunk_size_inst` is the standard handle for shrinking peak memory if other GPU users push us over budget — see `docs/dev/architecture/chunking.md`.

## Relationship to the nested solver

Both solvers converge to the same physical operating point; the choice is one of cost trade-offs:

| | Nested | FullJacobian |
|---|---|---|
| Per-iter math | block-2×2 wire Newton + coupled 2×2 V_clamp Newton | block-3×3 Thomas + boundary Schur, all unknowns in one Newton step |
| mid1 chip iter count (fp32) | `n_outer = 4, n_inner = 1` | `n_newton = 4` |

Benchmark on mid1 chip preset, inst=64, batch=64, 5 warmup + median of 5 timed runs (compile time excluded). Each dtype uses its own calibrated iteration counts under the current 2×2-outer scheme.

### fp64 — accuracy / verification path

| batch | solver | mode | latency [ms] | peak mem [GB] |
|---|---|---|---|---|
| 64 | nested (n_out=5, n_in=3) | eager | 3718 | 7.7 |
| 64 | nested (n_out=5, n_in=3) | torch.compile | 3050 | 5.4 |
| 64 | fulljac (n=5) | eager | 1724 | 16.5 |
| 64 | fulljac (n=5) | **torch.compile** | **1586** | 7.5 |

### fp32 — chip-preset production (current default)

| batch | solver | mode | latency [ms] | peak mem [GB] |
|---|---|---|---|---|
| 64 | nested (n_out=4, n_in=1) | eager | 1554 | 3.8 |
| 64 | nested (n_out=4, n_in=1) | torch.compile | 582 | **2.7** |
| 64 | fulljac (n=4) | eager | 813 | 8.2 |
| 64 | fulljac (n=4) | **torch.compile** | **319** | 3.8 |

### Cross-cutting observations

* **fp32 is ~5× faster + uses ~50% memory vs fp64** — the chip preset defaults to fp32 for that reason; fp64 is available for accuracy verification (`solver_calibrate.* --dtype float64`).
* **Nested wins on memory in both dtypes** — block-2×2 wire Jacobian uses ~4× less Jacobian storage than FullJacobian's block-3×3 plus V_X-in-vector overhead, and nested scales to larger batches where FullJacobian OOMs at fp64.
* **FullJacobian's lower fp32 latency does NOT make it a recommended production path.** The chip preset documents it as a reference/experimental implementation, NOT a routine swap-in. Its role is solver-correctness debug, FD Jacobian verification, and small-workload cross-check against nested. Production simulation uses nested.

**Status: debug / cross-check only.** `NestedSolver1T1R` is the production / golden default; `FullJacobianSolver1T1R` exists so the two can be cross-validated against each other in `test_full_jacobian_solver.test_full_jacobian_matches_nested`. Both solvers benefit from `torch.compile` (kernel fusion + dropped intermediates). For a one-off cross-check run, point `[xbar.core_config.solver_config]._neurox_type` at `FullJacobianSolver1T1RConfig`; do **not** treat that swap as a production fallback.
