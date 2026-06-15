# `neurox/xbar/_1t1r/nested_solver.py`

## Current role

`NestedSolver1T1R` is the **block-Gauss-Seidel + implicit-Newton** DC solver for a fabricated 1T1R tile. It is the default solver on the chip preset; `FullJacobianSolver1T1R` (which lifts every unknown into one Newton step) is the alternative when an independent reference is needed or when the full coupled linearization is preferred.

The legacy `NewtonRaphsonSolver1T1R` has been deleted: its Sherman-Morrison clamp coupling left a μA-scale `wire_bl` tail that this solver and FullJacobian both eliminate.

The solver decomposes the coupled multivariate nonlinear system into two well-conditioned sub-problems with unique solutions, eliminating the multi-equilibrium / rail-pseudo-fixed-point trap that the simultaneous Newton solver could fall into.

## Variables solved

For each cell at row `r`, column `c`:

- `V_BL[c, r]` — BL wire node voltage
- `V_SL[c, r]` — SL wire node voltage
- `V_X[c, r]` — internal node between RRAM and access-NMOS

Plus per-column boundary clamps:

- `V_BL_clamp[c]` — BL clamp voltage (TIA output)
- `V_SL_drive[c]` — SL drive voltage

`V_WL[r]` is an input (not solved).

## Algorithm

```
Phase 1  Warm start: Padé V_X seed + first-order IR-drop V_BL/V_SL seed +
         cell-sum-driven clamp seed (BL uses I_R, SL uses I_N).

Phase 2  Pre-loop cell refresh: one cell Newton at the IR-drop seed so
         the first outer step has correct (i_r, i_n, g_cell_eff).

Phase 3  Coupled outer Newton × n_outer (clamp-FIRST per outer iter):
  (a) Coupled 2×2 outer Newton step on (V_BL_clamp, V_SL_drive):
        - K_inner is a 2×2 matrix per col carrying BL/SL cross-coupling;
          computed by two block-2×2-tridiagonal solves with basis-vector
          RHSs (one per V_clamp component).
        - Outer Jacobian (2×2 per col) =
            [[r_bl·g_bl·(1−K[0,0])−1,  −r_bl·g_bl·K[0,1] ],
             [−r_sl·g_sl·K[1,0],       r_sl·g_sl·(1−K[1,1])−1]]
        - ΔV_clamp = clamp(−J⁻¹·F_outer, ±MAX_OUTER_STEP) via torch.linalg.solve.
  (b) Inner array Newton × n_inner at the NEW V_clamp:
        - cell Newton → V_X, I_R, I_N, g_cell_bl_eff, g_cell_sl_eff
        - coupled block-2×2 wire Newton via solve_block_tridiagonal,
          BL wire KCL uses I_R, SL wire KCL uses I_N
        - damped update v_BL/v_SL ← v + clamp(dv, ±MAX_INNER_STEP)

Phase 4  Exit-state cell refresh so the returned i_cell / V_X align with
         the returned (V_BL, V_SL); the clamp-first ordering means the
         wire and clamp are already mutually consistent.

Phase 5  Optional residual diagnostics (only when compute_residuals=True).
         BL wire residual uses I_R, SL wire residual uses I_N.
```

## Why this is correct

For our chip:

1. **Inner sub-problem (V_clamp fixed)** is a block-2×2 tridiagonal M-matrix-flavour system with positive-definite diagonal blocks. Each cell's KCL has monotone `I_NMOS(V_X)` vs `I_RRAM(V_BL − V_X)`, so `V_X` is unique per `(V_BL, V_SL)`. The coupled wire Newton therefore has a unique fixed point at any frozen clamp pair.

2. **Outer fixed point** on `(V_BL_clamp, V_SL_drive)` is the composition of the driver responses (TIA: monotone in `I_BL_port`; SL driver: monotone in `I_SL_port`) with the array response. The 2×2 outer Newton converges quadratically near the unique fixed point given the damping cap.

3. **Rail pseudo-equilibria** that trapped the simultaneous Newton solver are eliminated because the outer Newton operates per-column on a 2×2 problem whose Jacobian is well-conditioned away from rails. The damped Newton step cannot land in a rail-locked region unless the port currents are physically out of the driver's reachable range — and in that case the rail IS the correct physics.

## Iteration knobs (`NestedSolver1T1RConfig`)

| Field | Purpose | Chip preset value (fp32) |
|---|---|---|
| `n_outer` | Coupled 2×2 outer Newton iterations on `(V_BL_clamp, V_SL_drive)` | 4 |
| `n_inner` | Inner block-2×2 array Newton iterations per outer step | 1 |

Damping caps are method-intrinsic class constants (`MAX_OUTER_STEP__V = 0.10`, `MAX_INNER_STEP__V = 0.05`) — not exposed in the chip-preset TOML because they are not chip-tuned.

These knobs are **fixed numerical constants** (not non-ideality toggles), so they live in `NestedSolver1T1RConfig(Solver1T1RConfig)` — not in any Policy. Solvers have no Policy at all. The chip preset carries the config as a `[xbar.core_config.solver_config]` section with `_neurox_type = "NestedSolver1T1RConfig"`; `CircuitCore1T1R` reads it and dispatches via `Solver1T1R.from_config(config.solver_config, ...)`.

### Calibration methodology

Pick the values via `neurox.tools.solver_calibrate.nested`. The framework is **chip-parameter-free**:

1. **Primary**: step-ratio plateau detection. The tool sweeps candidate iteration counts, records the per-candidate solution `u_n`, then computes `step_n = max|u_n − u_{n-1}|`. When the ratio `step_n / step_{n-1}` crosses `ratio_threshold` (default 0.5) the iterate sequence has hit its fp round-off floor — further iterations only oscillate within numerical noise, so the candidate at the previous step is the smallest fully-converged choice.

2. **Sanity guard**: relative residual check. After plateau detection, each residual class is verified against a workload-derived denominator (`max|I_cell|` for cell + wire residuals, `max|V_BL_node|` for clamp residuals). The default `reltol = 1e-2` is a methodological constant — it sits safely above the fp32 accumulated round-off floor (~`ε_fp32 · sqrt(N_ops) · signal_scale ≈ 0.8%`) so the guard does not false-fire under fp32; fp64 workloads land many orders of magnitude below it.

3. **Staged 2-axis sweep**: Stage A pins `n_inner` at a generous value (default 5) and sweeps `n_outer`; Stage B pins `n_outer` at the picked value and sweeps `n_inner`. A small margin (`outer_margin = 1`, `inner_margin = 0`) is added.

The outer Newton is a **2×2 coupled** step on `(V_BL_clamp, V_SL_drive)` — implicit-function-theorem `K_inner` is a 2×2 matrix carrying the BL ↔ SL cell cross-coupling, derived by two block-2×2-tridiagonal solves with `e_0` basis RHSs. The outer Jacobian and Newton step both work in this 2×2 system, so a variable-SL driver is supported natively without algorithmic changes. (An SL-grounded chip has near-zero cross terms, so the 2×2 outer is numerically close to two independent 1D steps with a small algebra overhead.)

Calibrated at **fp32** (production dtype) on the mid1 chip preset, the step plateau lands at `n_outer = 3` and `n_inner = 1`. With margins the preset uses `(n_outer = 4, n_inner = 1)`. fp64 simulation can reuse the same counts — extra Newton steps cost nothing once the solver is at any floor, and fp64 residuals only get tighter.

## API contract

Same as `Solver1T1R.solve_dc`. Additionally exposes:

- `solve_array_fixed_clamp(*, v_bl_clamp__V, v_sl_drive__V, ...)` — runs only the inner array Newton loop with the supplied clamps frozen. Used by `solver_calibrate.nested` Stage 1 and by tests that want to inspect inner convergence in isolation.

## Compatibility

- `Solver1T1RDCOP` and `Solver1T1RResiduals` are shared with `FullJacobianSolver1T1R`; switching between solvers does not change the caller's data interface.
- `CircuitCore1T1RConfig.solver_config` carries either `NestedSolver1T1RConfig` or `FullJacobianSolver1T1RConfig`; the registry inside `Solver1T1R.from_config(...)` picks the matching class via `@Solver1T1R.register_key(...)`. The Policy tree has no solver field — solvers carry no Policy.
- Both chunk axes (`CircuitCore1T1RPolicy.solve_chunk_size_x` for A and `solve_chunk_size_inst` for B) are bit-exact under the nested solver in noise-off policies — verified by `tests/test_xbar_chunking.py::test_a_axis_bit_exact` and `::test_b_axis_bit_exact` (and the mixed-axis variants).
