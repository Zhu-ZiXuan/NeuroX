# `neurox/tools/solver_calibrate/`

## Purpose

Pick fixed iteration counts for every solver in the 1T1R DC family (`NestedSolver1T1R`, `FullJacobianSolver1T1R`, and the embedded `OpAmpTIA` inner Newton) so that the runtime path runs a `torch.compile`-friendly fixed-trip-count graph. Calibration is a one-shot off-line job — the chip preset stores the picked counts, and the production solver never monitors anything at runtime.

## Calibration methodology

The framework is **chip-parameter-free** by design — it never references ADC bits, ADC range, model output, or any other downstream concern. The picked count guarantees the solver has converged within its own iterate sequence's numerical floor.

### Step-ratio plateau detection (primary)

For each candidate iteration count `n`, the tool measures `u_n` (the solver's output) over the full workload. Then for each adjacent pair:

```
step_n = max |u_n − u_{n-1}|        # over batch / col / row / unknown class
ratio_n = step_n / step_{n-1}
```

Newton's method gives geometrically-decreasing steps during the convergent phase (ratio ≪ 1) and ratio → 1 when round-off dominates. The picked `n*` is the smallest where `ratio_{n*+1} > ratio_threshold` (default `0.5`) — meaning iter `n*+1`'s step is at least half as large as `n*`'s, signalling the floor.

The criterion uses only the iterate sequence's self-comparison. There is no "reference solution" and no chip-tuned absolute threshold — chip-dependent floating-point behaviour is baked into the floor itself, so the picker adapts naturally.

### Relative residual guard (sanity)

After the plateau pick, the residuals are verified against workload-derived signal scales:

```
|F_cell|.max / max|I_cell|     < reltol     (default 1e-2)
|F_wire|.max / max|I_cell|     < reltol
|F_clamp|.max / max|V_BL_node| < reltol
```

`reltol = 1e-2` (1%) is a methodological constant, NOT chip-tuned. It is set safely above the fp32 accumulated round-off floor (~`ε_fp32 · sqrt(N_ops) · signal_scale` ≈ 0.8% for our chip's 64-row wire ladder) so the guard does not false-fire under fp32, while still catching genuine divergence (a 1% residual ratio means wire KCL is off by 1% of cell current — clearly broken). fp64 workloads land 8+ orders of magnitude below this threshold.

If a chip's workload pushes wire ladders much longer or its signal scale much smaller, the operator may need to raise `--reltol` further. The CLI exposes `--reltol`, `--ratio-threshold`, and `--margin` for that.

### Why not absolute residual / ADC-relative / huge-iter reference?

* **Absolute residual (`< 1 nA`)** changes meaning per chip — a chip with 10 μA operating current sees 1 nA as 100 ppm, while a chip with 100 μA sees it as 10 ppm. The same threshold is too tight for some chips, too loose for others.
* **ADC-relative (Δv vs V_LSB)** conflates solver accuracy with ADC quantization. A solver that's intrinsically wrong but lucky enough that the error rounds to the same ADC code would "pass". Tightening the ADC then exposes the masked solver error. We calibrate the solver alone here; ADC has its own calibration.
* **Huge-iter reference (`u_n` vs `u_huge`)** has a chicken-and-egg problem — to declare the reference trusted, one must check it doesn't change at `u_huge + δ`, which is itself a plateau check. The reference adds expense for no extra signal.

The plateau detector resolves this cleanly: convergence is defined by the iterate sequence's own behaviour, not by any external comparison.

## Default dtype: fp32

All three CLI tools default to **fp32** (`--dtype float32`). Rationale:

* Production simulation (LeNet / BERT inference, training, throughput evaluations) runs in fp32 for speed and memory. The chip preset's calibrated iteration counts must match the production dtype.
* fp32 plateau is reached in fewer iterations than fp64 — the round-off floor is higher, so further iterations bring no benefit. fp32-calibrated counts are correct (and sufficient) for fp32 simulation.
* The same counts also work in fp64 — extra Newton iterations beyond the fp32 floor cost almost nothing once the solver is at any floor, and the residuals only get tighter.
* fp64 is available via `--dtype float64` for accuracy verification or debug.

The chip preset comments record the production dtype explicitly so future calibration runs can be cross-checked.

## Modules

- `_plateau.py` — pure plateau-detection + residual-guard logic. `CandidateRow`, `WorkloadScale`, `PickResult` data containers; `pick_iter_by_step_ratio`, `check_residual_relative_guard`, `pick_with_plateau_and_guard` functions.
- `_common.py` — xbar builder, sweep aggregator (`aggregate_xbar_sweep`) that streams a workload through a list of candidate `Solver1T1R` instances and computes per-candidate step deltas + residuals.
- `tia.py` — CLI for `OpAmpTIA.n_newton`. Single-axis sweep on V_clamp.
- `nested.py` — CLI for `(n_outer, n_inner)` of `NestedSolver1T1R`. Staged 2-axis sweep (Stage A: pin `n_inner`, sweep `n_outer`; Stage B: pin `n_outer`, sweep `n_inner`).
- `full_jacobian.py` — CLI for `FullJacobianSolver1T1RConfig.n_newton`. Single-axis sweep on the full unknown vector.

## CLI common surface

Every calibrator is **config-driven** — workload, sweep, criteria,
and reproducibility knobs (dtype, seed) live in TOML. CLI carries
only:

- `--config` (required) — run TOML.
- `--device` (optional) — execution device; defaults to CPU when omitted (no implicit GPU pickup).
- `--plot-dir` (optional) — matplotlib output for the step / residual curves.
- `--log-level` (default `INFO`).

Per-run TOML schema is per-tool (`solver_calibrate_{tia,nested,full_jacobian}.toml`
under `example/config/`):

- `[xbar]` — chip preset via `_neurox_use`.
- `[workload]` — sampling sweep dimensions; the nested / full_jacobian
  flavours also require `inst_shape == [batch_w]`.
- `[sweep]` — candidate iteration counts + `ratio_threshold` / `reltol` /
  margin(s). The nested calibrator carries the extra Stage-A pin
  `inner_ref` and separate `outer_margin` / `inner_margin`.
- `[runtime]` — `dtype` (`"float32"` / `"float64"`) and `seed`.

## Output

Each tool prints:

1. Per-candidate metrics: iter count, step magnitude vs previous, residual maxima.
2. The pick reason (plateau detected, where, residual guard ratios).
3. A TOML fragment ready to paste into `[xbar.core_config.solver_config]` or `[xbar.core_config.tia_config]`.

## Adding a new solver family

For a `Solver1T1R`-based family, write a CLI module modelled on `full_jacobian.py`: build per-candidate `Solver1T1R` instances, call `aggregate_xbar_sweep`, and forward the resulting rows + scale to `pick_with_plateau_and_guard`. The plateau detector and residual guard are family-agnostic.

For a solver that doesn't return `Solver1T1RDCOP` (e.g. a standalone driver like `OpAmpTIA`), write a custom sweep aggregator modelled on `tia.sweep_n_newton` that wraps the family's outputs into `CandidateRow` instances, then reuse `pick_with_plateau_and_guard`.
