# 1T1R Circuit Solver — Design and Strategy

This document describes the DC solver that computes the equilibrium
operating point of a 1T1R RRAM crossbar tile in
[`neurox/xbar/solver_1t1r.py`](../../neurox/xbar/solver_1t1r.py). It is
meant to be read end-to-end before anyone modifies the solver, adds a
new xbar variant, or tries another optimization. It explains **what**
the solver does, **how** the current implementation avoids nested
loops, and — just as important — **why** several theoretically-
attractive alternatives were evaluated and rejected.

**State-ownership note** — per
[`temp/state_holding.md`](../../temp/state_holding.md), per-VMM
device state lives inside the
:class:`~neurox.device.RRAM` /
:class:`~neurox.device.NMOS` /
:class:`~neurox.analog.ClampDriver` modules.  The solver
consumes three opaque per-VMM **runtime-state dataclasses**
(`rram_snapshot`, `nmos_snapshot`, `clamp_snapshot`) sampled once by
`Core1T1R.forward` and threaded unchanged through every internal
helper.  Older versions of this doc described "raw mismatch tensors"
in the `solve(...)` signature; that contract has been replaced by
the runtime-state contract.

## 1. Problem Statement

A 1T1R cell is the series stack

```
v_bl ── RRAM ── v_x ── Switch ── v_sl
```

where the RRAM is a nonlinear resistor (sinh I–V) and the Switch is
a MOSFET in triode acting as a gate-controlled resistor. An
`Xbar1T1R` tile contains `col_num × row_num` such cells arranged in a
grid. Each column shares a BL wire terminated at an ADC-side driver;
each row shares an SL wire terminated at an SL driver; each row also
has a WL signal that gates all switches on that row.

Both wires are modeled as resistor ladders — every inter-cell
segment has resistance `r_step`, and the driver-to-first-cell segment
has resistance `r_first`. Cell currents drawn off the ladder cause IR
drop along the wire, so the voltage at a cell deep in the array is
lower than at the driver end.

The solver must compute, for given driver voltages, WL logic signals,
and programmed conductances:

* `v_bl[batch, col, row]` — BL wire voltage at every node,
* `v_sl[batch, col, row]` — SL wire voltage at every node,
* `v_x[batch, col, row]`  — intermediate node voltage at every cell,

such that **Kirchhoff's Current Law** holds at every wire node and
every cell-internal node simultaneously. The equilibrium state is the
unique solution of a coupled system of nonlinear equations:

* **Cell KCL** (per cell, scalar in `v_x`):

  ```
  f_cell(v_x) = g_sw(v_bl, v_sl, v_x) · (v_x − v_sl)
              − RRAM.i(v_bl − v_x, g_rram)   = 0
  ```

* **BL wire KCL** (per BL node along `dim=-1`):

  ```
  i_driver_into_node − Σ i_wire_leaving_node = i_cell_at_node
  ```

* **SL wire KCL** (per SL node along `dim=-2`): symmetric.

The output the macro asks for is the **driver-port current** per BL
and per SL — a scalar per column and per row — plus the **dynamic BL
clamp voltage** `v_bl_clamp` (per column) and the **TIA output
voltage** `v_out` (per column) produced by the non-linear TIA
black-box on the BL side.

### 1.0 Definition domain

The 1T1R DC solver is defined on:

* **Strictly-positive finite wire resistances on a 1-D ladder** —
  every entry of `bl_wire.segment_r__MOhm` / `sl_wire.segment_r__MOhm`
  must be `> 0`.  `Wire.fabricate` enforces this; per-segment values
  are supplied directly from `Core1T1RConfig`'s
  `bl_r_driver_to_first__MOhm` / `bl_r_cell_to_cell__MOhm` /
  `sl_r_driver_to_first__MOhm` / `sl_r_cell_to_cell__MOhm` fields
  (no length × resistivity inside `Wire`).  Ideal-wire / zero-
  resistance topologies are out of scope; future strap / bridge /
  comb source-line networks need a separate solver since the
  current Jacobian relies on the tridiagonal-plus-rank-1 structure.
  See `temp/wire.md` for the per-segment wire encoding and the
  `sl_topology` discussion.
* **`sl_topology = "row_shared"` only** — `Core1T1RConfig`
  exposes the SL topology as an explicit field; the current solver
  is built around the row-shared SL (SL extends along the row axis
  like the WL).  `sl_topology = "col_shared"` is reserved for a
  future build and is rejected up at `Core1T1R.fabricate`
  (`NotImplementedError`).
* **Tiles with `col_num > 1` and `row_num > 1`** — the KCL residual
  uses `torch.diff` along both wire axes and the wire-conductance
  ladder needs at least one inter-node segment per wire.  Single-
  row / single-column tiles are not physically interesting either.
  `XbarConfig.__post_init__` enforces this at the config layer, and
  `solve()` re-checks the runtime shape on entry as a defensive
  guard for direct callers.

### 1.1 Dynamic BL boundary: rank-1 TIA fold-in

The BL clamp is **not** a fixed voltage source.  A non-linear TIA
(`neurox.analog.tia.TIA`) sits at every column's BL driver port and
sets the clamp voltage according to the closed-loop equation

```
v_out         = opamp_gain · (v_ref − v_bl_clamp)
i_bl(column)  = nmos.ids(v_g=v_nmos_bias, v_d=v_out, v_s=v_bl_clamp,
                         β, V_th)
```

where `i_bl(column) = i_cell.sum(dim=-1)`.  Because the cell current
itself depends on the BL clamp, the array solve and the TIA solve
are coupled.  We resolve the coupling **inside the BL Newton step**:
`TIA.solve_dc` returns both the converged `(v_bl_clamp, v_out)`
*and* the small-signal sensitivity `dVblClamp_dIbl`, and the BL
wire correction folds that sensitivity into the tridiagonal
Jacobian as a rank-1 update via Sherman-Morrison.

Concretely — linearising the TIA at the current operating point
gives `Δv_drive ≈ dVclamp/dI · Σ_k g_eff_k · Δv_bl_k`, which only the
`j = 0` BL wire residual depends on (`∂R_0/∂v_drive = −1/r_first`).
The augmented wire Jacobian is therefore `J_aug = J_orig + u v^T`
with `u_j = (−dVclamp_dI / r_first) · δ_{j,0}` and `v_k = cell_g_eff_k`.
Sherman-Morrison gives the correction in two tridiagonal solves
(one for `-R → z`, one for `u → y`) plus a scalar combination:

```
Δv_bl = z − y · (v^T z) / (1 + v^T y)
```

When the TIA is rail-clipped, `dVclamp/dI = 0` masks `u` to zero and
the rank-1 term vanishes — the BL step collapses to the plain
tridiagonal solve.  Ideal-BL (zero-resistance) topologies are out of
scope for this solver; the wire is required to be a strictly-positive
finite resistor ladder (`segment_r__MOhm > 0` everywhere).

The outer Newton loop is (per-VMM runtime snapshots are sampled once
by `Core1T1R.forward` and threaded as
`rram_snapshot` / `nmos_snapshot` / `clamp_snapshot` — see
`temp/state_holding.md`):

```
for outer_iter in range(N_UNROLL_OUTER):
    i_cell, cell_g_eff, v_x_node = _solve_cell_newton_warm_start(
        v_bl_node, v_sl_node, wl_drive_grid__V, v_x_node,
        rram_snapshot, nmos_snapshot)
    bl_clamp__V, r_clamp_in__MOhm = clamp_driver.solve_clamp(
        i_cell.sum(-1), clamp_snapshot, v_clamp_init__V=bl_clamp__V)
    f_bl_kcl = col_wire_kcl_residual(v_bl_node, bl_clamp__V.unsqueeze(-1), …)
    f_sl_kcl = row_wire_kcl_residual(v_sl_node, sl_drive__V, …)
    dv_bl_node = _solve_bl_newton_clamp_rank1(
        v_bl_node, f_bl_kcl, g_cell_eff, r_clamp_in__MOhm)
    dv_sl_node = _solve_sl_newton_tridiagonal(
        v_sl_node, f_sl_kcl, cell_g_eff)
    v_bl_node, v_sl_node = v_bl_node + dv_bl_node, v_sl_node + dv_sl_node
```

Cross-call locality is exploited twice over: (a) the cell Newton
seeds `v_x` from the previous outer iteration's converged value
(skipping the Padé divider after the first call), and (b) the TIA
Newton seeds `v_clamp` from the previous outer iteration's
`v_bl_clamp`.  Both contract from a near-converged starting point,
so the fixed iteration counts (1 cell Newton step, 4 TIA Newton
steps) buy a lot of headroom.

See §6.2 for the implementation and §7.2 for the convergence
analysis.

The TIA itself runs an internal 4-iteration 1D Newton on its single
unknown `v_clamp` with an analytical derivative (no finite-difference
artefacts).  See `signal_chain.md` for the device-level details.

## 2. Tensor Shapes and Conventions

* Cell tensor shape: `[*batch, col_num, row_num]`.
* BL wire runs along `dim = -1` (the row axis); `col_num` independent
  BLs, each `row_num` nodes long. Driver sits at index 0.
* SL wire runs along `dim = -2` (the col axis); `row_num` independent
  SLs, each `col_num` nodes long. Driver sits at index 0.
* Cell current sign convention: `i_cell > 0` when current flows from
  BL to SL through the cell (normal read direction).
* `bl_d` has shape `[*batch, col_num, 1]` (broadcast to all row nodes
  of each BL); `sl_d` has shape `[*batch, 1, row_num]` (broadcast to
  all col nodes of each SL).

`neurox/xbar/solver.py` contains the shared helpers that operate on
these shapes: `solve_tridiagonal`, `col_wire_kcl_residual`,
`row_wire_kcl_residual`, `col_driver_current`, `row_driver_current`.
The four wire helpers all consume a `segment_g__uS` tensor (length =
number of wire nodes) produced by `Wire.fabricate`; per-segment
conductance is encoded at index `[0]` (driver segment) and indices
`[1:]` (inter-node segments).  The tridiagonal Jacobian templates
(self-conductance diagonal and inter-node off-diagonal) are not
cached on the wire: they are derived inside the solver from
`segment_g__uS` because they are solver-specific representations of
the same physical state.

## 3. Design Constraints

The solver is invoked from inside HAT-training forward passes, so:

1. **Speed is dominant.** The solver runs tens of thousands of times
   per training epoch on full BERT / vision macros.
2. **`@torch.compile` friendliness.** The hot path must fuse cleanly
   into Triton kernels without graph breaks.
3. **No graph breaks inside autograd.** Although the solver itself
   runs under `torch.no_grad()` (the autograd path uses an STE float
   surrogate — see `HATLinear.forward`), its outputs feed into
   compiled regions of the operator, so avoid `.item()`, Python
   control flow on GPU tensors, and in-place mutation where possible.
4. **Correctness at the ADC level.** The solver's output is fed to
   the ADC's `torch.bucketize` call, which is a step function. Any
   per-node residual below the ADC's bucket width is invisible in
   the final codes, so we target **code-exact** output (not
   bit-exact raw currents).
5. **Bit-exact against the analytic reference on the ideal-wire
   regression.** `scripts/bench_xbar_error.py` compares every column
   of 1000 random inputs against a 12-step analytic Newton reference
   and must report 100 % exact-match on `ideal_1t1r`. Any change
   that breaks this is a regression.
6. **Memory is not a bottleneck.** Peak solver footprint on the
   largest test case (batch=1024, full unrolled flat path) is ~1 GB
   — trivially fits on a single 97 GB GPU.

## 4. The Classical Two-Level Newton Solver (baseline)

The original implementation (kept conceptually as variant `A` in
`scripts/xbar_profile/solver_variants.py`) is a textbook
Schur-complement Newton:

```text
for outer_iter in range(MAX_ITER):                       # damped Block-Jacobi
    for each cell:
        for vx_iter in range(VX_NEWTON_ITER):            # inner scalar Newton
            v_x -= f_cell(v_x) / f_cell'(v_x)
        i_cell, g_cell_eff = compute(v_bl, v_sl, v_x)
    R_bl, R_sl = KCL(v_bl, v_sl, i_cell)
    if max|R| < I_ATOL: break
    d_bl = Thomas(diag = g_cell_eff + g_wire,  rhs = −R_bl, dim=-1)
    d_sl = Thomas(diag = g_cell_eff + g_wire,  rhs = −R_sl, dim=-2)
    v_bl, v_sl = v_bl + 0.8·d_bl, v_sl + 0.8·d_sl
```

### Why it works

* **Schur elimination**: the cell-level variable `v_x` is eliminated
  analytically at every outer iter, so the outer system is just the
  wire voltages. The Schur-complement conductance
  `g_cell_eff = g_rram_diff · g_sw / (g_rram_diff + g_sw)` captures
  how the cell current linearises at the wire level.
* **Block-Jacobi**: BL and SL wire systems are solved independently
  each iteration. Each decouples into a standalone tridiagonal (the
  BL wire touches only BL nodes of the same column; the SL wire
  touches only SL nodes of the same row), which the Thomas algorithm
  solves in `O(N)` per wire.
* **Damping 0.8**: because the Block-Jacobi decomposition discards the
  cross-coupling `∂V_BL/∂V_SL`, the outer Newton becomes linearly-
  convergent; damping prevents oscillation.

### Why it's slow

Three nested loops:

1. Inner `v_x` Newton: `VX_NEWTON_ITER = 4` steps of `sinh + cosh` per
   cell, per outer iter. 4 × N_outer = up to 80 evaluations per call.
2. Outer Newton: up to `MAX_ITER = 20` iterations with damping 0.8,
   typically converging in 7.
3. Each outer iter calls `_cell_solve` **twice** — once for the
   residual check, once for the step — so the inner 4-step Newton
   runs **twice per outer iter**.

Per-call cost is dominated by (a) sinh/cosh evaluations and (b)
Thomas sweeps (unrolled to 64 straight-line steps by Inductor).

## 5. Design Space — Methods We Evaluated

The file [`1t1r_solver.md`](../../1t1r_solver.md) has several
LLM-authored discussions of possible optimizations. We implemented
every concrete method from those discussions as a standalone variant
in `scripts/xbar_profile/solver_variants.py` and benchmarked each
head-to-head (see §8). The methods are:

### 5.1 State-space augmentation (variant B)

Promote `v_x` to a first-class solver state; do exactly one Newton
step on `v_x` per outer iter. `v_x` tracks the wire voltages as they
converge.

* **Kills**: the inner 4-step Newton inside each outer iter.
* **Retains**: the 4-step Newton in the warm-start phase (still needs
  an accurate first `i_cell` to compute residuals); the Python outer
  loop; damping.
* **Effect**: per-iter sinh/cosh work drops 4× with no outer-iter
  count change.

### 5.2 Zero-wire-resistance warm start (variant D)

Solve the r=0 problem first (`v_bl = bl_d, v_sl = sl_d` everywhere —
one `_cell_solve`), then derive first-order IR-drop wire voltages
analytically:

```
S_bl[k] = Σ_{j≥k} i_cell[j]          (reverse cumsum on dim=-1)
v_bl[0]   = bl_d − S_bl[0]·r_first
v_bl[k+1] = v_bl[k] − S_bl[k+1]·r_step
```

This puts the initial `(v_bl, v_sl)` inside the quadratic-convergence
basin of the outer Newton.

* **Kills**: nothing structurally, but *shortens* the outer loop
  from ~7 to ~5 iterations.
* **Retains**: the inner 4-step Newton; the Python outer loop.

### 5.3 Anderson acceleration (variant C)

Type-II Anderson with depth `m = 4`: keep the last `m` pairs
`(G(X_k), F(X_k) := G(X_k) − X_k)` of the outer fixed-point map and
apply a `[B, m, m]` least-squares correction at every iter. This
recovers superlinear convergence without building the full Jacobian.

* **Kills**: nothing structurally, but supposed to reduce outer iters
  from ~7 to ~3.
* **Added cost**: `[m, B, N]` history tensors and a batched LS solve
  per iter.

### 5.4 Padé closed-form V_X (variant E)

For the scalar cell KCL let `u = v_bl − v_x` and `y = v_bl − v_sl`.
The first-order Taylor approximation `sinh(α u)/α ≈ u` makes the KCL

```
g_sw · (y − u) = g_rram · u
```

which gives the **linear (Ohmic) estimate**

```
u₀ = g_sw · y / (g_sw + g_rram)
```

One Newton step on the **true** sinh KCL from `u₀` drops the error
quadratically: for α = 0.5 and `y ≤ 0.2 V` the linear estimate has
~0.1 % error, so one Newton step gives ~1e-7 in `v_x` — two orders
of magnitude tighter than the legacy 4-step Newton from the
midpoint guess.

* **Kills**: the inner Newton loop entirely (replaced by `u₀` +
  1 Newton step). No Python `for` loop left anywhere.
* **Added cost**: one extra `cosh` evaluation for `g_cell_eff`
  (already present in the baseline); otherwise strictly fewer
  evaluations (1 sinh + 1 cosh vs. 4 sinh + 4 cosh).

The approximation is a **Taylor** expansion, not Padé in the strict
sense, but we use "Padé" as shorthand throughout the discussion
because the linear estimate serves the same role as a Padé [0,0]
rational seed.

### 5.5 Undamped Newton (variant H)

Set `DAMPING = 1.0`. Damping was necessary to prevent oscillation
when the initial state was far from the fixed point (Block-Jacobi
loses convergence information). With the warm-started state already
inside the quadratic basin, damping is no longer needed.

* **Kills**: nothing, but shortens the outer loop from ~5 to ~3
  iterations.

### 5.6 Fixed-depth unroll (variant G)

Replace the dynamic Python loop

```python
for _ in range(MAX_ITER):
    ...
    if res.item() < tol: break
    ...
```

with a compile-time unrolled loop of fixed depth `N_UNROLL_OUTER`:

```python
for _ in range(N_UNROLL_OUTER):  # unrolled at @torch.compile time
    ...
```

no `if`, no `.item()` sync. Running N unconditional iterations is
fine even when the solver has already converged — subsequent
iterations apply zero-magnitude Newton steps.

* **Kills**: the Python outer loop, the CPU-GPU `.item()` sync, and
  the per-iter kernel-launch overhead (every iter is inlined into
  one giant Triton kernel).
* **Added cost**: always runs `N_UNROLL_OUTER` iterations regardless
  of convergence; unrolling multiplies the compiled kernel size by
  `N_UNROLL_OUTER`.

### 5.7 Methods we did NOT implement (and why)

* **Full-block MNA with batched banded solver**. Builds the complete
  `(V_BL ⊕ V_SL ⊕ V_X)` Jacobian and inverts it in one LU /
  block-tridiagonal solve. Recovers true quadratic Newton convergence
  → 2–3 outer iters instead of 4. Needs a **batched** banded solver;
  PyTorch has no native op for this, cuSOLVER has one but requires a
  C++ extension + `cudaGraph` compatibility work. Weeks of
  engineering for maybe 1.5× further speedup. Deferred.
* **Custom Triton kernel for Thomas (Cyclic Reduction / PCR)**. The
  current Thomas implementation is Inductor-unrolled into a 64-step
  straight-line kernel. CR/PCR would reduce this to `log₂(64) = 6`
  parallel steps. Likely 1.3–1.5× speedup on the Thomas portion,
  but requires a custom Triton kernel. Deferred.
* **Padé [1,1] rational** `sinh(x)/x ≈ (1 + x²/6)/(1 − x²/12)`.
  For our default `α = 0.5, y ≤ 0.2 V`, the simpler linear-estimate
  * 1 Newton (above) already gives ~1e-7 — no benefit from the
  rational form. Would help for larger α.
* **JFNK / GMRES**. Matrix-free Krylov with matrix-vector products
  via finite differences. Each outer iter needs ≥2 extra forward
  evaluations for the `J · v` approximation, which dwarfs the 1-iter
  savings from recovering quadratic convergence at our iteration
  counts. Not cost-effective for a 5-iter loop.

## 6. The Shipped Solver — Fully-Flat GU Variant

The solver currently in `neurox/xbar/solver_1t1r.py` combines:

* **Warm start** (§5.2) for the wire voltages.
* **Padé closed-form V_X** (§5.4) for the bootstrap call; the cell
  Newton inside the outer loop is seeded from the previous
  iteration's `v_x` instead of a fresh divider seed.
* **Fixed-depth unroll** (§5.6) — `N_UNROLL_OUTER = 5`.
* **Undamped Newton** (§5.5) — `damping = 1.0`.
* **Rank-1 TIA fold-in** (§1.1) — `TIA.solve_dc` returns
  `dVclamp_dI__MOhm` alongside the converged voltages; the BL wire
  Newton folds this into its Jacobian via Sherman-Morrison.
* **Reused TIA warm starts** — the previous outer iteration's
  `v_bl_clamp` is threaded into the next `solve_clamp` call so the
  TIA's internal Newton contracts from a near-converged seed.
* **Geometry-free construction** — the solver is an `nn.Module`
  bound only to the device modules and the two fabricated
  :class:`~neurox.device.Wire` instances (``bl_wire`` / ``sl_wire``).
  ``num_col`` / ``num_row`` are read from
  ``rram_snapshot.state_g__uS.shape`` inside :meth:`solve`; every
  per-segment tensor (``segment_r__MOhm`` / ``segment_g__uS``) is
  read straight off the wires, and the tridiagonal Jacobian
  templates are derived locally from ``segment_g__uS`` (see
  ``temp/wire.md``).  Static programmed RRAM conductance
  (used for the warm-start divider) is read directly from
  ``self.rram.state_g__uS``; the per-VMM noisy snapshot
  (``rram_snapshot.state_g__uS``) drives every real cell evaluation
  inside the Newton loop.  See ``temp/1t1r_solver.md``.

### 6.1 Structure

```
neurox/xbar/solver_1t1r.py

helpers (private, Dynamo-inlined):
  _solve_cell_pade_warm_start    ← Padé divider seed + 1 Newton step
                                    + final eval (cold start)
  _solve_cell_newton_warm_start  ← 1 Newton step from a known-near
                                    v_x_node_init + final eval
                                    (reuses prev outer iter's v_x_node)
  _solve_cell_newton_once        ← shared body of the two above
  _solve_bl_newton_clamp_rank1   ← Thomas tridiagonal +
                                    Sherman-Morrison rank-1 fold-in
                                    of the clamp driver's
                                    r_clamp_in__MOhm
  _solve_sl_newton_tridiagonal   ← plain Thomas tridiagonal on SL

solver attributes (geometry owned by the fabricated wires):
  self.bl_wire                   ← fabricated BL Wire; exposes
                                    segment_r__MOhm / segment_g__uS.
  self.sl_wire                   ← fabricated SL Wire (same surface).

solve()-local intermediates:
  rram_state_g_static__uS        ← static programmed RRAM conductance
                                    expanded to execution shape;
                                    used only by the warm-start
                                    divider (kept distinct from the
                                    noisy snapshot).

public:
  NewtonRaphsonSolver1T1R.solve  — single entry point; runs the
                                    warm start, the unrolled outer
                                    Newton, and the driver-port
                                    extraction in one compiled region.
```

### 6.2 The flat finite-wire kernel

`solve` (the single compiled entry point) executes, in one fused
region:

1. **Warm start phase 1**: seed `v_bl_node = bl_clamp_ref__V` (TIA's
   static reference), run
   `i_cell_init, _, v_x_node_init = _solve_cell_pade_warm_start(
   v_bl_node_seed, sl_drive_grid__V, …)`.  Linear-estimate V_X + 1
   Newton step + final eval — no inner loop.  The returned
   `v_x_node_init` carries the cell operating point into the outer
   loop.
2. **Clamp warm-start prediction**: `bl_clamp__V, _ =
   clamp_driver.solve_clamp(i_cell_init.sum(-1), clamp_snapshot)`.
   The first entry of the returned tuple becomes `bl_clamp__V`;
   the second entry (the small-signal sensitivity) is unused on the
   first call because the Sherman-Morrison fold-in only enters from
   iteration 1 onward.  The call cold-starts from the driver's
   static op (no `v_clamp_init__V`).
3. **Warm start phase 2**: first-order IR-drop `(v_bl_node, v_sl_node)`
   via reverse-cumsum + cumsum along each wire axis (named
   `i_bl_downstream` / `i_sl_downstream` and their prefix sums),
   with `bl_clamp_grid__V` as the BL boundary.
4. **`N_UNROLL_OUTER = 5` unrolled outer Newton iterations**, each:
   * `i_cell, cell_g_eff, v_x_node = _solve_cell_newton_warm_start(
     v_bl_node, v_sl_node, wl_drive_grid__V, v_x_node,
     rram_snapshot, nmos_snapshot)` — one Newton step from the
     previous outer iteration's `v_x_node`, skipping the Padé
     divider.
   * `bl_clamp__V, r_clamp_in__MOhm = clamp_driver.solve_clamp(
     i_cell.sum(-1), clamp_snapshot, v_clamp_init__V=bl_clamp__V)` —
     refresh the dynamic boundary and its sensitivity, warm-starting
     the TIA Newton from the previous `bl_clamp__V`.
   * KCL residuals `f_bl_kcl` (against the latest `bl_clamp_grid__V`)
     and `f_sl_kcl`.
   * `_solve_bl_newton_clamp_rank1` — Thomas tridiagonal for the
     original BL system (giving `dv_bl_base`) + a second Thomas
     solve for the `v_rank1_rhs` column vector (only non-zero at
     `j = 0`) + a per-(batch, col) scalar Sherman-Morrison combine.
     The rank-1 fold-in is bypassed when the TIA is rail-clipped
     (`r_clamp_in__MOhm = 0`) and when the BL is ideally clamped
     (both wire resistances zero).
   * `_solve_sl_newton_tridiagonal` — plain Thomas tridiagonal on SL.
   * Undamped `v_bl_node`, `v_sl_node` update.
5. **Driver-port currents** `i_bl_driver` / `i_sl_driver` from final
   `v_bl_node`, `v_sl_node`, `i_cell`, using the converged
   `v_bl_clamp_grid__V`.  `SolverResult` exposes these all under
   quantity-first names — `i_bl_driver`, `i_sl_driver`,
   `v_bl_node`, `v_sl_node`, `v_x_node`, `i_cell`, `v_bl_clamp`,
   `v_out`.

Wire self-conductance diagonals are not built inside the kernel —
they are buffers on the solver `nn.Module`, pre-computed at
construction and `.view`-broadcast at the use site.

### 6.3 What's killed and what remains

| loop | before | after |
|---|---|---|
| Inner V_X Newton (per-cell, per outer iter) | 4 iter | **0** (Padé) |
| Warm-start V_X Newton | 4 iter | **0** (Padé) |
| Outer wire Newton | Python `while`, up to 20 iter, with `.item()` sync | **0** (fixed-depth unroll, 4 iters) |
| Ideal-wire fast path V_X | 4 iter | 4 iter (unchanged, Inductor unrolls at compile time) |

The **finite-wire path has zero dynamic Python loops** — from
Dynamo's perspective the whole solver is a single straight-line
graph. The ideal-wire fast path is already a single kernel call with
Inductor-unrolled inner work; we kept the legacy 4-step Newton there
because it is the historically bit-exact reference that the
regression bench (`scripts/bench_xbar_error.py`) compares against.

### 6.4 Why the ideal-wire path is separate

When every wire segment length is zero, the equilibrium wire voltages
collapse to the driver voltages and no outer Newton step is needed —
a single `_cell_solve` plus column/row sums gives the driver
currents.  This solver does **not** carry that branch: ideal-wire
topologies are routed through a separate ideal-xbar path, and the
1T1R solver requires `segment_r__MOhm > 0` everywhere (enforced
inside `Wire.fabricate`).

## 7. Accuracy Analysis

### 7.1 Padé V_X accuracy

Let `x = α·u` be the argument of `sinh` in the cell KCL. The linear
estimate `u₀` satisfies the **Ohmic** KCL (first-order Taylor). One
Newton step on the **true** sinh KCL from `u₀` cuts the residual
quadratically. Concretely:

* **Default `α = 0.5`, `y ≤ 0.2 V`** → `|x| ≤ 0.1`, so
  `|sinh(x)/x − 1| < 0.17 %`. Linear estimate `u₀` has ≤ 0.1 %
  relative error in `v_x`. One Newton step drops it to ~1e-7.
* **`α = 1.0` (a more nonlinear device)** → `|x| ≤ 0.2`, Taylor
  error ~0.67 %. One Newton step drops to ~5e-5 — still well below
  the ADC step (1.6e-2 mA).
* **`α = 2.0` (extreme)** → `|x| ≤ 0.4`, Taylor error ~2.7 %. Would
  need a 2nd Newton step. Not in our regime.

For the shipped bundled configs (`ideal_1t1r.toml`, `default_1t1r.toml`,
and every BERT / LeNet example) this is strictly more accurate than
the legacy 4-step midpoint Newton.

### 7.2 Convergence of the outer loop

With the warm start putting us in the quadratic basin and damping
removed, the outer Newton on `(V_BL, V_SL)` converges in **3
iterations on typical inputs** (empirically observed). The shipped
`N_UNROLL_OUTER = 4` provides a 1-iter safety margin.

Sweeping `N_UNROLL_OUTER ∈ {3, 4, 5, 6, 8}` on the
`bench_xbar_error.py` regression all give **100 % bit-exact match
with the analytic reference** — even `N = 3` produces ADC codes
indistinguishable from the 12-step reference.

We ship with `N = 4` for production safety without sacrificing
measurable speed: 4 is 1 iter above observed convergence, 3 is
zero-margin. For harder device configurations (higher `α`, more
aggressive wire resistance, larger arrays) `N_UNROLL_OUTER` is the
one knob a user would adjust.

### 7.3 Cross-variant error (measured at batch=100, `default_1t1r`)

Max absolute BL driver-current error vs. the baseline iterative
solver A (itself at `I_ATOL = 1e-6` per node):

| variant | max \|Δbl_i\| (mA) | % of ADC step (0.0164 mA) |
|---|---:|---:|
| A (Layer 0 reference) | 0.00e+00 | 0.00% |
| B (state aug) | 1.43e-08 | 0.00009% |
| C (Anderson) | 1.85e-05 | 0.11% |
| D (warm start) | 2.95e-05 | 0.18% |
| E (Padé) | 2.95e-05 | 0.18% |
| H (undamped BD) | 1.95e-05 | 0.12% |
| **GU (shipped)** | **1.87e-05** | **0.11%** |

All within one per-node `I_ATOL × N` ceiling (`N · I_ATOL = 6.4e-5`
mA for 64 rows). All **≈500× below the ADC quantization step**, so
the ADC bucketize produces identical codes for every variant.

The shipped GU variant shows the smallest drift of all non-trivial
methods — on par with B (which runs the exact 4-step inner Newton)
despite using only the 1-step Padé V_X. This is because the
undamped + warm-started outer iteration converges more tightly than
the damped one (no overshoot near the fixed point).

## 8. Performance

All measurements: `cuda:1`, float64, 20 timed calls after 5 warmup,
`default_1t1r` with full non-ideality (deterministic — noise
commented out), `batch=100` = 100 random VMM samples per call.

### 8.1 Head-to-head (batch=100)

| variant | per_call (ms) | per_sample (µs) | outer iters | peak memory | vs A | vs BD | loops remaining |
|---|---:|---:|---:|---:|---:|---:|---|
| A (Layer 0) | 7.85 | 75.0 | 7 | 72 MB | 1.00× | 0.57× | outer (Py) + inner (4) |
| B | 5.79 | 57.9 | 7 | 82 MB | 1.36× | 0.78× | outer (Py) + inner (1) |
| C | 20.7 | 207 | 6 | 379 MB | 0.38× | 0.22× | outer (Py) + inner (4) + AA LS |
| D | 5.69 | 57.0 | 5 | 87 MB | 1.38× | 0.79× | outer (Py) + inner (4) |
| BD (prev on-disk) | 4.50 | 45.0 | 5 | 96 MB | 1.74× | 1.00× | outer (Py) + inner (1) |
| E | 4.64 | 46.4 | 5 | 83 MB | 1.69× | 0.97× | outer (Py) |
| H | 2.95 | 29.5 | 3 | 96 MB | 2.66× | 1.53× | outer (Py) + inner (1) |
| ED | 2.84 | 28.4 | 3 | 83 MB | 2.76× | 1.58× | outer (Py) |
| **GU (shipped)** | **2.31** | **23.1** | 4 (fixed) | 101 MB | **3.40×** | **1.95×** | **none** |

### 8.2 Scaling with batch

| variant | b=1 | b=16 | b=100 | b=1024 |
|---|---:|---:|---:|---:|
| A (baseline) | 7.47 ms | 7.43 ms | 7.85 ms | 51.83 ms |
| BD (prev on-disk) | 3.12 ms | 3.14 ms | 4.50 ms | 28.96 ms |
| **GU (shipped)** | **2.00 ms** | **2.01 ms** | **2.31 ms** | **14.26 ms** |

The flat kernel's compile is ~15 s on first call (per unique shape);
subsequent calls hit the cached kernel and run at the numbers shown.

### 8.3 N_UNROLL_OUTER sensitivity

At batch=100 / batch=1024 on the shipped GU solver:

| N_UNROLL_OUTER | b=100 | b=1024 | correctness (64 k codes) |
|---:|---:|---:|---|
| 3 | 1.56 ms | 9.65 ms | 100 % bit-exact |
| **4 (shipped)** | **2.31 ms** | **14.27 ms** | **100 % bit-exact** |
| 5 | 2.98 ms | 18.92 ms | 100 % bit-exact |
| 6 | 3.68 ms | 22.97 ms | 100 % bit-exact |
| 8 | 5.10 ms | 31.55 ms | 100 % bit-exact |

Essentially linear scaling of wall time with `N_UNROLL_OUTER` — each
unrolled iteration costs ≈ 0.75 ms at batch=100, ≈ 4.75 ms at
batch=1024.

### 8.4 Reproducing the numbers

```bash
# GPU availability check
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader

# Regression (must show 100.00 % on both configs)
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python scripts/bench_xbar_error.py

# Single-solver timing (what 8.2 uses)
PYTHONPATH=. python -m scripts.xbar_profile.bench_nonideal

# Full head-to-head of all 12 variants (what 8.1 uses)
PYTHONPATH=. python -m scripts.xbar_profile.bench_variants
```

## 9. @torch.compile Strategy

Two compile regions only:

### 9.1 `_ideal_wire_solve` (ideal-wire fast path)

One kernel. Takes driver voltages and cell tensors, produces
driver-port currents. Contains `_solve_cell_midpoint` (the 4-step V_X Newton
unrolled into straight-line code by Inductor), plus column/row
sum reductions. No Python loops at runtime. Called whenever all
four wire resistances are 0.

### 9.2 `_flat_finite_wire_solve` (finite-wire flat path)

One kernel. Contains:

* the entire warm start (`_solve_cell_pade` + cumsum/reverse-cumsum IR-drop),
* `N_UNROLL_OUTER = 4` copies of the per-iter physics (one cell_pade,
  two KCL residuals, two tridiagonal Thomas solves, two voltage
  updates) fused together,
* the final driver-port extraction.

Inductor's traces across the unrolled iterations:

* CSE the wire self-conductance pattern (constant across iters).
* CSE `g_sw = switch.g_sw__mS(wl, g_on)` (constant).
* CSE `bl_d`, `sl_d` (constants).
* Fuse each iter's Thomas sweep with the next's cell_pade.

The resulting Triton kernel is ~4× larger than a single iter but
launches once instead of 5–20 times. Kernel-launch overhead
(≈ 5 µs each) × 5 = 25 µs saved, plus 5 `.item()` CPU-GPU syncs
(~10 µs each) = 50 µs saved. For small batches (b = 1) those 75 µs
are 4 % of the per-call time; for large batches the fusion
benefits through reduced HBM traffic are more significant — at
b = 1024 the GU path is 2× faster than BD despite running 4 outer
iters vs 5.

### 9.3 `dynamic=True` everywhere

The solver is invoked at many shapes (different batch sizes, tile
counts, and — for other xbar backends — potentially different cell
shapes). `dynamic=True` lets Inductor recompile only when the
**shape pattern** changes, not for every new concrete shape.

### 9.4 Avoiding graph breaks

`.item()` is **not called** in `solve()` on the finite-wire path —
the flat kernel runs `N_UNROLL_OUTER` iterations unconditionally.
On the ideal-wire path it's never called at all. This means a
higher-level compiled region (e.g., the HAT operator's forward)
can absorb `solve()` into its own compile graph without graph
breaks.

## 10. Invariants and Correctness Guarantees

1. **Output shape**: `(i_bl_driver, i_sl_driver)` are
   `[*batch, col_num]` and `[*batch, row_num]` respectively.
2. **Output dtype**: matches the input dtype (float32 or float64).
3. **Output device**: matches the input device (CPU or CUDA).
4. **Ideal-wire regression**: the
   `scripts/bench_xbar_error.py::ideal_1t1r` test compares 1000
   random (w, x) pairs against a 12-step Newton analytic reference
   and reports 100 % bit-exact on every column of every input.
5. **Unit tests**: `python -m pytest test/` must report
   `35 passed`.
6. **Ideal-wire fast path is bypassable**: any test that wants to
   exercise the finite-wire path must pass `Wire` objects to the
   `Xbar1T1R` constructor; absence of `Wire` = ideal wire path.

## 11. Extension Guide

If you add a new xbar topology (e.g., 1S1R, 1T1MTJ, 1T1C) you will
**not** reuse this solver directly — its wire-ladder and
Schur-complement structure are specific to a 1T1R cell. But the
principles generalize:

### 11.1 Methods that are topology-agnostic

* **Zero-parasitic warm start**: always works. Solve the cheap
  (r = 0) version, derive first-order IR-drop from cumsum-based
  formulae matched to your cell topology.
* **Closed-form / Padé cell solve**: if your cell KCL admits a
  low-order polynomial or rational approximation for typical
  operating points, use it as an initial guess + 1 Newton step.
  Skip only if the device is sufficiently nonlinear (large `α`
  equivalent, steep `I-V`) that multiple Newton steps are needed
  for accuracy.
* **Fixed-depth unroll**: always works when (a) the outer Newton
  has a known worst-case convergence iter count from physics and
  (b) extra unconditional iterations are cheap (no side effects).
* **Undamped Newton with warm start**: always worth trying once
  you have a physics-grounded warm start; benchmark vs damped.

### 11.2 Methods that are worth re-trying for different topologies

* **Anderson acceleration** pays off when outer iter count is large
  (say, ≥ 10). If your new topology has stiffer IR-drop or more
  complex cross-coupling and the outer Newton needs many more iters
  than 1T1R's 3–5, Anderson may become net-positive even with its
  `[m, B, N]` history overhead.
* **State augmentation** only matters if your inner Newton loop
  is expensive per-cell (nonlinear `sinh`, `exp`, etc.). For
  topologies with purely linear cells there's no inner Newton to
  collapse.

### 11.3 Structure to follow

Put the new solver at `neurox/xbar/solver_<topology>.py` and
expose a class with the same shape contract as
`CrossbarNewtonRaphsonSolver.solve`: input driver voltages + cell
tensors + wire resistances, output `(bl_i, sl_i)` driver currents.

Reuse the shared helpers in `neurox/xbar/solver.py` — they are
topology-agnostic. Add topology-specific helpers (your cell KCL,
your Schur complement) alongside the new solver file.

Copy the test style from `scripts/bench_xbar_error.py` — a
12-step analytic reference is a cheap regression gate; add it for
your topology and wire it into the bench matrix.

The scripts in `scripts/xbar_profile/` are **designed for reuse** —
they measure time and memory of any `Xbar*` class plugged in. Add a
new build-xbar function for your topology and the existing bench
harness will accept it.

## 12. Open Follow-ups

Not yet implemented, in rough priority order:

1. **Custom PCR Thomas kernel**. The biggest remaining on-hot-path
   cost is the Thomas sweeps (≈ 40 % of per-iter time at large
   batch). A Triton kernel doing `log N` parallel cyclic reduction
   would speed Thomas by ~1.3–1.5×. Engineering cost: a week to
   write + validate.
2. **Padé [1,1] rational for higher-α devices**. If a user ships a
   device model with `α > 1`, the current Taylor-based linear
   estimate degrades. Swapping to a rational approximation
   `sinh(x)/x ≈ (1 + x²/6)/(1 − x²/12)` extends the accurate range
   to `α ≤ 2`. Small code change, needs a validation sweep.
3. **Adaptive `N_UNROLL_OUTER`** at compile time. For specific
   device configs where convergence is particularly fast, allow
   the macro factory to set a tile-specific `N_UNROLL_OUTER` on
   the `Xbar1T1RConfig` rather than the module-level constant.
   Minor ergonomics change; no new algorithm.
4. **Full MNA with batched cuSOLVER banded solver**. Recovers true
   quadratic Newton; ~1.5× over the shipped flat solver. Needs a
   C++ extension for the cuSOLVER call with `cudaGraph` integration
   — weeks of work.

## 13. Glossary

* **Cell KCL** — the scalar equation `g_sw·(v_x − v_sl) =
  rram.i(v_bl − v_x, g_rram)` at every 1T1R cell. Solvable for
  `v_x` given `v_bl, v_sl`.
* **Wire KCL** — the continuity equation at each wire node: current
  into node (from driver + neighboring wire segments) = current out
  (to the cell injecting at that node).
* **Schur complement** — the effective per-cell conductance
  `cell_g_eff = rram.g_diff · nmos.g_d / (rram.g_diff + nmos.g_d)`
  seen by the wire Jacobian after eliminating `v_x_node`. Used as
  the diagonal of the tridiagonal wire system.
* **Warm start** — initial guess for the iterative solver,
  typically derived from a cheaper approximate problem.
* **Padé / Taylor closed-form** — here, the linear first-order
  approximation `sinh(α·u)/α ≈ u` plus one Newton step on the true
  KCL.
* **N_UNROLL_OUTER** — the fixed depth of the unrolled outer
  Newton loop in the shipped flat finite-wire solver. 4 on disk;
  3 is the minimum observed convergence plus safety margin.
* **GU** — the variant name in `scripts/xbar_profile/solver_variants.py`
  corresponding to the shipped flat finite-wire solver: Padé
  V_X + fixed-depth unroll + undamped Newton.
