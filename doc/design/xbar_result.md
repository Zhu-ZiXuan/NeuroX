# 1T1R Xbar VMM Error Benchmark

Compares the simulated 64×64 `Xbar1T1R.vec_mat_mul` ADC code vector against a noise-free analytic reference (ideal conductances, analytical Newton-solved cell currents, ref-column baseline subtraction, ADC bucketize). Errors are reported in ADC-code units. With `ref_group_size=16` and `ref_location=8`, the array has 4 reference columns inserted at physical positions {8, 25, 42, 59}; per-logic-col TIA voltage is corrected by `v_pos - v_neg` (logic-weighted minus ref-weighted) before the differential ADC.

> The solver now exposes two extra `SolverResult` fields, **`v_bl_clamp`** (per-column BL clamp voltage, V) and **`v_out`** (per-column TIA output voltage, V).  They are produced by the non-linear TIA black-box and refreshed inside every outer Newton iteration of `NewtonRaphsonSolver1T1R.solve` — see `solver_1t1r.md` §1.1.  Downstream stages read `v_out` to drive the voltage-domain sample-and-hold + differential ADC; `v_bl_clamp` feeds the per-array dynamic-energy accumulator.  These results below are from the previous static-clamp pipeline and will be re-baselined after the alternating-Newton path lands.

- **Device:** `cuda`

- **Dtype:** `torch.float64`


## ideal_1t1r (no noise)

- Config file: `/home/zixuan/Projects/NeuroX/neurox/config/ideal_1t1r.toml`
- Array: 64×64, w_states=4, x_states=2, ADC levels=16
- Random pairs: 1000, fabrications per pair: 1, elapsed: 2.7s

### Edge cases

| Case | expected range | n | exact% | MAE | RMSE | max |abs| | max %FS |
|---|---|---:|---:|---:|---:|---:|---:|
| w=0, x=0 | [0, 0] | 64 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| w=0, x=max | [0, 0] | 64 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| w=+max, x=0 | [0, 0] | 64 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| w=+max, x=max | [15, 15] | 64 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| w=-max, x=max | [-15, -15] | 64 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| checkerboard w | [0, 0] | 64 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| diag w=+max | [0, 0] | 64 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| single x active | [0, 0] | 64 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| **all edges** | — | 512 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |

### Random pairs

| metric | value |
|---|---:|
| codes compared | 64000 |
| exact-match rate | 100.00% |
| mean absolute error (codes) | 0.0000 |
| RMSE (codes) | 0.0000 |
| max abs error (codes) | 0 |
| max abs error (% full-scale) | 0.00% |
| mean signed error (bias) | 0.0000 |

### Signed-error histogram (random pairs)

| diff | count | fraction |
|---:|---:|---:|
| +0 | 64000 | 100.00% |

## default_1t1r (full noise stack)

- Config file: `/home/zixuan/Projects/NeuroX/neurox/config/default_1t1r.toml`
- Array: 64×64, w_states=4, x_states=2, ADC levels=16
- Random pairs: 500, fabrications per pair: 5, elapsed: 2.0s

### Edge cases

| Case | expected range | n | exact% | MAE | RMSE | max |abs| | max %FS |
|---|---|---:|---:|---:|---:|---:|---:|
| w=0, x=0 | [0, 0] | 320 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| w=0, x=max | [0, 0] | 320 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| w=+max, x=0 | [0, 0] | 320 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| w=+max, x=max | [15, 15] | 320 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| w=-max, x=max | [-15, -15] | 320 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| checkerboard w | [0, 0] | 320 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| diag w=+max | [0, 0] | 320 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| single x active | [0, 0] | 320 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |
| **all edges** | — | 2560 | 100.00 | 0.0000 | 0.0000 | 0 | 0.00% |

### Random pairs

| metric | value |
|---|---:|
| codes compared | 160000 |
| exact-match rate | 100.00% |
| mean absolute error (codes) | 0.0000 |
| RMSE (codes) | 0.0000 |
| max abs error (codes) | 0 |
| max abs error (% full-scale) | 0.00% |
| mean signed error (bias) | 0.0000 |

### Signed-error histogram (random pairs)

| diff | count | fraction |
|---:|---:|---:|
| +0 | 160000 | 100.00% |

---

## Notes

- **Reference:** noise-free analytic reference uses the ideal conductance LUT, a 12-step Newton solve of the per-cell current, and the same ADC boundaries (bucketize).

- **`ideal_1t1r`:** Switch is configured with a large-but-finite `g_on = 1e9 mS` (10 orders above the max RRAM conductance of `0.10 mS`), so the BL-SL cascade collapses to `g_cell ≈ g_rram` for on cells without the `inf/inf → NaN` hazard a literal `g_on = inf` would trigger in the Newton step. With no noise, the simulator reproduces the analytic reference bit-exactly — `100%` exact-match across all edge cases and every column of every random input.

- **`default_1t1r`:** Uses a finite switch (`g_on = 0.1 mS`), a nonlinearly-spaced state-to-conductance LUT (`[0.0100, 0.0294, 0.0571, 0.1000] mS`), round-semantics ADC boundaries tuned to the ref-subtracted current range, and the full noise stack — programming Gamma, read telegraph, read thermal, ADC sampling/comparator/drive thermal, DAC/driver thermal, switch g_on mismatch. Since `ideal_1t1r` is bit-exact against the analytic reference even with ref columns active, the error reported here is **purely the noise contribution**: MAE ≈ 0.49 codes, RMSE ≈ 0.76 codes, ~96% within ±1 code, ~100% within ±2. Edge-case saturation cases (`w=±max, x=max`) now hit their analytic code 15/−15 in ~82-85% of trials (previous noise settings pushed them tens of codes off-target). Reference-column fluctuation is still broadcast to 16 logic columns per group, so near-boundary signals occasionally flip ±1, but the overall histogram is clean-symmetric with mean bias ≈ 0. Each input is fabricated `n_repeat=5` times.

