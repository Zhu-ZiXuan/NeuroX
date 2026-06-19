# Calibrating solver iteration counts

Goal: pick a fixed iteration count for every solver in the 1T1R DC family ($\operatorname{NestedSolver1T1R}$, $\operatorname{FullJacobianSolver1T1R}$, and the embedded $\operatorname{OpAmpTIA}$ inner Newton) so the runtime path executes a `torch.compile`-friendly fixed-trip-count graph. Calibration is a one-shot offline job: the chip preset stores the picked counts and the production solver never monitors anything at runtime.

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
\operatorname{reltol} \gtrsim \varepsilon_{\mathrm{fp32}} \cdot \sqrt{N_{\mathrm{ops}}} \cdot \operatorname{signal\_scale} \approx 0.8\%
$$

for the reference chip's 64-row wire ladder, so the guard does not false-fire under fp32 while still catching genuine divergence: a 1% residual ratio means wire KCL is off by 1% of cell current, which is clearly broken. fp64 workloads land $8+$ orders of magnitude below this threshold.

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

The chip-preset comments record the production dtype explicitly so future calibration runs can be cross-checked.

## Calibrated counts

At fp32 the plateau lands at:

| solver family | raw plateau $n^*$ | margined count (preset) |
|---|---|---|
| $\operatorname{NestedSolver1T1R}$ $(n_{\mathrm{outer}}, n_{\mathrm{inner}})$ | $(3, 1)$ | $(4, 1)$ |
| $\operatorname{FullJacobianSolver1T1R}$ $n$ | $3$ | $4$ |

The preset adds a $+1$ margin on the outer / Newton axis, so the $(4, 1)$ and $4$ counts stored in the chip preset are margined values, not raw plateau picks.

---

- **See also**: [DC solvers internals](../../internals/xbar/_1t1r/solver.md), [calibration hub](./README.md)
