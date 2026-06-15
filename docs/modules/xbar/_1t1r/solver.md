# `neurox/xbar/_1t1r/solver.py`

## Current role

This module owns the 1T1R-family DC-solver framework. Three things live here:

1. **`Solver1T1RConfig`** — empty marker base for concrete solver configs. Subclasses (`NestedSolver1T1RConfig`, `FullJacobianSolver1T1RConfig`) carry only the workload-tuned iteration counts; pure algorithmic safety constants (Newton damping caps, Jacobian floors) belong on the concrete solver class as `ClassVar` attributes.

2. **`Solver1T1R`** — abstract base with a registry that maps config types to concrete solver classes. Each concrete solver decorates itself with `@Solver1T1R.register_key(SomeSolverConfig)` so `Solver1T1R.from_config(config=...)` can dispatch without an explicit `isinstance` ladder.

3. **`Solver1T1RDCOP` / `Solver1T1RResiduals`** — shared DC operating-point + residual containers used by every 1T1R solver flavour. Keeps the upstream `CircuitCore1T1R` API solver-agnostic.

## Why per-family base (not generic `Solver`)

The earlier draft had a generic `Solver` base in `xbar/solver.py` accepting `**kwargs` for device handles. That base was leaky: every concrete topology family (1T1R, future 2T2R, differential, …) has a different fixed set of boundary actors and thus a different `from_config` signature. The 1T1R-specific base spells the four mandatory actors (`rram`, `nmos`, `bl_driver`, `sl_driver`) right in the method signature — `**kwargs` is gone, type-checking works.

Other topology families should define their own `SolverXxx` base in their own module rather than try to subclass `Solver1T1R`.

## Solvers have no Policy

`Solver1T1R` has no companion `SolverPolicy`. Pure numerical methods have no per-source nonideality toggle: every knob is either a workload-tuned constant (carried in `Solver1T1RConfig` subclasses) or a method-intrinsic safety bound (on the concrete solver class). The composite `CircuitCore1T1RPolicy` therefore has no `solver` field.

## Residual container

`Solver1T1RResiduals` carries five per-(batch, col[, row]) absolute residual tensors:

- `cell__uA` — `|I_NMOS − I_RRAM|` per cell
- `wire_bl__uA` / `wire_sl__uA` — wire KCL residual per node
- `clamp_bl__V` / `clamp_sl__V` — `|TIA(I_PORT) − V_clamp|` per column

Populated only when the caller passes `compute_residuals=True` to `solve_dc`; the hot path leaves the whole bundle as `None` so the extra evaluations get elided from the compiled graph. The wire-residual side-products of the inner-only `NestedSolver1T1R.solve_array_fixed_clamp` entry point fill `clamp_*__V` with zeros (clamps are pinned inputs in that flow, not solved variables).

## Calibration

Each concrete solver has a matching `solver_calibrate.<name>` tool. The common framework (see `neurox/tools/solver_calibrate/_plateau.py`) is **chip-parameter-free**:

* **Primary criterion** — step-ratio plateau detection. For each candidate iteration count `n`, the tool measures `step_n = max|u_n − u_{n-1}|` over the full workload. When the ratio `step_n / step_{n-1}` exceeds `ratio_threshold` (default `0.5`), the iterate sequence has reached its fp round-off floor — further iterations cannot improve `u`. The smallest such `n` (minus 1, plus a margin) is the recommended count.
* **Sanity guard** — relative residual. After the pick, each residual class is verified against a workload-derived denominator: current residuals against `max|I_cell|`, voltage residuals against `max|V_BL_node|`. The default `reltol = 1e-2` (1%) is a methodological constant set safely above the fp32 accumulated round-off floor (`~ε_fp32 · sqrt(N_ops) · signal_scale ≈ 0.8%` for our chip's 64-row wire ladder) while still catching genuine divergence. fp64 workloads land 8+ orders of magnitude below this threshold.

Both criteria avoid any chip-specific absolute threshold — they are purely about whether the solver's own iterate sequence has stopped progressing, judged by self-comparison. ADC quantization, model-level accuracy, and any other downstream concerns are out of scope here.
