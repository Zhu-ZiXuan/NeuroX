# Calibrating solver iteration counts

Goal: pick a fixed iteration count for every fixed-trip-count Newton in the 1T1R DC path — the array solver $\operatorname{NestedParallelRailSolver}$ ($n_{\mathrm{outer}}$, $n_{\mathrm{inner}}$), the embedded $\operatorname{OpAmpTia}$ inner Newton ($n_{\mathrm{newton}}$), and the per-cell access-node condensation $\operatorname{XbarCell1t1rDetail}$ ($n_{\mathrm{newton}}$) — so the runtime path executes a `torch.compile`-friendly fixed-trip-count graph. Calibration is a one-shot offline job: the chip preset stores the picked counts and the production solver never monitors anything at runtime. Each Newton has its own calibrator package — `neurox.tools.calibrate_solver` (array solver), `neurox.tools.calibrate_tia` (TIA inner Newton, CLI `python -m neurox.tools.calibrate_tia._opamp`), and `neurox.tools.calibrate_cell` (per-cell condensation).

The framework is **chip-parameter-free** by design. It never references ADC bits, ADC range, model output, or any other downstream concern. The picked count guarantees the solver has converged within the numerical floor of its own iterate sequence.

## Step-ratio plateau detection (primary)

For each candidate iteration count $n$, the tool measures $u_n$ (the solver output) over the full workload. Then for each adjacent pair, taking the maximum over batch / col / row / unknown class:

```text
step_n  = max |u_n - u_{n-1}|
ratio_n = step_n / step_{n-1}
```

In LaTeX:

$$
\operatorname{step}_n = \max \lvert u_n - u_{n-1} \rvert,
\qquad
\operatorname{ratio}_n = \frac{\operatorname{step}_n}{\operatorname{step}_{n-1}} .
$$

Newton's method gives geometrically decreasing steps during the convergent phase ($\operatorname{ratio} \ll 1$) and $\operatorname{ratio} \to 1$ once round-off dominates. The picked $n^*$ is the smallest $n$ where

$$
\operatorname{ratio}_{n^*+1} > \operatorname{ratio\_threshold}
\qquad (\text{default } 0.5),
$$

meaning iteration $n^*+1$'s step is at least half as large as $n^*$'s step, signalling the floor.

The criterion uses only the iterate sequence's self-comparison. There is no reference solution and no chip-tuned absolute threshold: chip-dependent floating-point behaviour is baked into the floor itself, so the picker adapts naturally.

## Relative-residual guard (sanity)

After the plateau pick, the residuals are verified against workload-derived signal scales:

$$
\frac{\max \lvert F_{\mathrm{cell}} \rvert}{\max \lvert I_{\mathrm{cell}} \rvert} < \operatorname{reltol},
\qquad
\frac{\max \lvert F_{\mathrm{wire}} \rvert}{\max \lvert I_{\mathrm{cell}} \rvert} < \operatorname{reltol},
\qquad
\frac{\max \lvert F_{\mathrm{clamp}} \rvert}{\max \lvert V_{\mathrm{BL\,node}} \rvert} < \operatorname{reltol},
$$

with default $\operatorname{reltol} = 10^{-2}$ (1%).

$\operatorname{reltol}$ is a methodological constant, **not** chip-tuned. It is set safely above the fp32 accumulated round-off floor

$$
\operatorname{reltol} \gtrsim \varepsilon_{\mathrm{fp32}} \cdot \sqrt{N_{\mathrm{ops}}} \cdot \operatorname{signal\_scale}
$$

so the guard does not false-fire under fp32 while still catching genuine divergence: a 1% residual ratio means wire KCL is off by 1% of cell current, which is clearly broken. fp64 workloads land $8+$ orders of magnitude below this threshold.

If a chip's workload pushes wire ladders much longer or its signal scale much smaller, the operator may need to raise `--reltol` further. The CLI exposes `--reltol`, `--ratio-threshold`, and `--margin` for that.

## Why not absolute residual / ADC-relative / huge-iteration reference?

- **Absolute residual ($< 1$ nA)** changes meaning per chip. A chip with 10 uA operating current sees 1 nA as 100 ppm, while a chip with 100 uA sees it as 10 ppm. The same threshold is too tight for some chips and too loose for others.
- **ADC-relative ($\Delta v$ vs $V_{\mathrm{LSB}}$)** conflates solver accuracy with ADC quantization. A solver that is intrinsically wrong but lucky enough that the error rounds to the same ADC code would pass; tightening the ADC then exposes the masked solver error. We calibrate the solver alone here, and the ADC has its own calibration.
- **Huge-iteration reference ($u_n$ vs $u_{\mathrm{huge}}$)** has a chicken-and-egg problem. To declare the reference trusted, one must check it does not change at $u_{\mathrm{huge}} + \delta$, which is itself a plateau check. The reference adds expense for no extra signal.

The plateau detector resolves this cleanly: convergence is defined by the iterate sequence's own behaviour, not by any external comparison.

## Default dtype: fp32

All three CLI tools default to **fp32** (`--dtype float32`). Rationale:

- Production simulation (LeNet / BERT inference, training, throughput evaluations) runs in fp32 for speed and memory, so the chip preset's calibrated iteration counts must match the production dtype.
- The fp32 plateau is reached in fewer iterations than fp64 because the round-off floor is higher, so further iterations bring no benefit. fp32-calibrated counts are correct and sufficient for fp32 simulation.
- The same counts also work in fp64: extra Newton iterations beyond the fp32 floor cost almost nothing once the solver is at any floor, and the residuals only get tighter.
- fp64 is available via `--dtype float64` for accuracy verification or debug.

The chip-preset comments record the production dtype explicitly so calibration runs can be cross-checked.

## From plateau to stored count

Each picked count is the raw plateau $n^*$ plus a fixed $+1$ safety margin — the nested solver takes the margin on its outer axis ($n_{\mathrm{outer}}$, inner left at its plateau), and the cell takes it on its single axis ($n_{\mathrm{newton}}$). The margined values are written into the chip config, never into this guide: the array-solver counts live in `[cim_macro.array_config.solver_config]` (`n_outer`, `n_inner`) and the per-cell count in `[cim_macro.array_config.cell_config]` (`n_newton`). To read a chip's counts, open its config; to re-pick them for a chip, re-run the tools below. This guide states the method and the margin rule, not any chip's numbers, because those go stale against the config.

## Per-cell condensation count

The per-cell access-node condensation in $\operatorname{XbarCell1t1rDetail}$ runs its own fixed Newton on the internal-node KCL $F_{\mathrm{X}} = I_{\mathrm{N}} - I_{\mathrm{R}}$ after a Pade current-divider seed; its `n_newton` is a separate calibrated knob owned by the Detail cell config (`[cim_macro.array_config.cell_config].n_newton`), not by the array-solver config. It is picked with the **same** step-ratio-plateau criterion as above, applied to the per-cell internal node $V_{\mathrm{X}}$ and the per-cell internal-KCL residual $\lvert F_{\mathrm{X}} \rvert$ rather than to the wire / clamp unknowns. It is calibrated by `neurox.tools.calibrate_cell`, a scheme-agnostic package separate from the array-solver calibrator (`neurox.tools.calibrate_solver`) and the TIA calibrator (`neurox.tools.calibrate_tia`): the per-cell condensation is the cell's responsibility, and a new cell type with a different internal topology calibrates its own count without touching the other calibrators. The run config carries the Detail cell fragment directly (a `cell_config` table, typically pulled from a scheme's chip params via `_neurox_use`) plus the grid / sweep / runtime sections. The tool sweeps `n_newton` over a representative operating grid — $v_{\mathrm{BL}}$ / $v_{\mathrm{SL}}$ across the read-voltage range, the word line off and on, and every programmed RRAM state — and reads the plateau at the worst point of that grid. The seed lands inside the Newton basin, so the per-cell plateau is reached in very few steps and $V_{\mathrm{X}}$ reaches its round-off floor quickly; the margined count it emits is written to `[cim_macro.array_config.cell_config].n_newton`.

One run emits **two TOML fragments** into `--output-dir`, in addition to the log:

- `cell_detail_n_newton.toml` — the margined `n_newton` pick to merge into the scheme's Detail cell fragment;
- `cell_linear.toml` — a complete linearized-cell (`XbarCell1t1rLinearConfig`) fragment: the shared physical fields copied from the input Detail config, and per-(state, WL-level) secant sub-conductances of the Detail cell extracted at the nominal operating point given by the grid's `v_bl_op__V` / `v_sl_op__V`, with the WL on/off threshold at the midpoint of the grid's two WL levels. Degenerate (cut-off) entries floor at a tiny positive conductance with a logged warning. Selecting the Linear cell is a pure config choice — point the array's `cell_config` table at the fragment via `_neurox_use`.

```bash
python -m neurox.tools.calibrate_cell._1t1r \
    --config <scheme_run_config>.toml \
    --device cpu \
    --output-dir <output_dir>
```

---

- **See also**: [DC solvers internals](../../internals/primitive/xbar/solver.md), [calibration hub](./README.md)
