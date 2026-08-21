# ye2023jssc validation results

> **PENDING RECAL — every number on this page is STALE.** The array's capacitive ledger is one total per cell NODE and the WL / BL drives ride their own converter seats, so the `array` row (and with it the array pin, the C_WL solve and its saturation claim, and the whole per-block power table) no longer describes what the code bills; the per-vector `.bl_cap` channel is retired, the BL conduction branch bills across the BL driver rail, and two converter rows (`wl_dac`, `bl_dac`) join the report. The gates run RED until the recalibration campaign re-solves the free set; this page is kept as the record of the PREVIOUS model, not as a current result. The free set is now four numbers — the two RS-CSA knobs and the two transcribed seat powers; the C_WL knob rides the array's WL node total and the per-code converter energies are unsolved zeros. The transfer, `i_tbl` and code gates are untouched — the solve and the lookup are bit-identical by construction — but they have not been re-run either.

Outcome of `make validate_ye2023jssc`, whose defaults are the record run: `--device cuda --n-w 64 --n-x 256 --repeat 8 --solve-chunk 4096 --seed 0`. Each of the 8 rounds programs a FRESH 64-die weight ensemble onto one macro built at `inst_shape=(64,)` and drives 256 fresh input vectors through ONE broadcast `vec_mat_mul`, so a sparsity point reads 8 x 256 x 64 x 64 = 8,388,608 output accesses. Both points replay the same seed — common random numbers, the same weight sequence and the same input uniforms thresholded per point — so the two points differ by their input threshold alone. The macros run float32 against a float64 closed-form oracle. Gate targets and reference targets are read from `anchors.toml`; params from `params.toml`.

The campaign has two parts. Five HARD GATES decide pass/fail — the harness exits non-zero unless all five pass. The dual-caliber power report is NOT gated: each Fig.19 point is fully consistent with the model under exactly one mounting hypothesis for the measured setup, and no single mounting explains both, so the two readings are reported side by side instead of one being fitted.

Four gates are INDEPENDENT — no free parameter is solved against the quantity they check. The RS-CSA energy gate is a CALIBRATION-CONSISTENCY check: `mirror_scale` and `e_fixed_per_op__fJ` are solved against exactly its two constraints, so it asserts that solve is jointly feasible and holds across both workloads, not that the model predicts the 470 fJ anchor.

## Calibration inventory

Exactly five numbers are free. Three carry `[calibrated]` in `params.toml`, two are `[transcribed]` paper block powers; every other field carries a non-free provenance tag and is pinned by it. Calibration inputs are an ENSEMBLE, not one draw: every residual below is quoted with its draw counts (`n_w` dies x `n_x` vectors x `repeat` rounds), never with a single seed.

| Parameter | Value | Solved against | Bound | Residual |
|---|--:|---|---|---|
| `c_wl__fF` (the C_WL knob) | 1.504 | the 50% caliber-Y array pin (`array` + `.bl_cond` + `.bl_cap` vs 56.150 uW) | C_WL in [60, 200] fF/row | C_WL = 200.0 fF/row, the UPPER EDGE; array pin -4.08% (53.858 uW) over `n_w` 64 x `n_x` 256 x `repeat` 8 |
| `mirror_scale` | 0.1446 | jointly: 470 fJ/conversion within +-10% at BOTH sparsity points AND per-code spread in [1.3, 1.8] | b = 16 * mirror_scale in [2.120, 2.742] | b = 2.3136; -8.7% / +8.1% flat, spread 1.319x |
| `e_fixed_per_op__fJ` | 392.5 | (the same constraint pair) | (the same band) | (the same) |
| `mux_driver_config.leakage_per_inst__uW` | 5.63 | Fig.19 Mux & Driver block power, flat across both points | — | 1.00x at both points |
| `timing_ctrl_config.leakage_per_inst__uW` | 14.03 | Fig.19 Timing & Mode Ctrl block power, flat across both points | — | 1.00x at both points |

**The C_WL knob is SATURATED.** Closing the 50% array pin exactly on the record draw would need about 620 fF/row — 200 plus the 2.29 uW deficit over the pin's C_WL slope `v_wl_sel^2 / T_AC` = 0.36/66 uW per fF/row, an affine split verified empirically to 1e-7 on a perturbed-`c_wl` run. The sanctioned band tops out at 200 fF/row, so C_WL sits at the bound edge and the remaining pin residual is reported rather than zeroed. That unclamped figure is itself draw-sensitive — half-scale ensembles (`repeat` 2) put it at 442 and 550 fF/row, about 20% either side — but every draw lands far outside the band, so the saturation conclusion is draw-robust. There is no headroom left: any later drift in `.bl_cond` (a change to `g_cell_on`, to the radix path, or to the weight-sparsity convention) moves the residual with nothing to absorb it.

**Residuals are ensemble readings, and the deficit is not draw noise.** Every round redraws both sides — 64 weight matrices and 256 input vectors — so a residual is a mean over 8 x 256 x 64 accesses, and the round-to-round spread of the model total is 44.517 +- 0.188 uW at 87.5% and 94.969 +- 0.202 uW at 50% (std / sqrt(rounds)). Within a round the standard error is measured by a crossed two-way decomposition into input-side and weight-side components, checked against the closed-form input-side prediction (0.777% predicted vs 0.794% measured at the 50% point). The dominant channel `.bl_cond` carries 0.79% SE at the 50% point (0.78% input-side, 0.08% weight-side) and 2.07% at 87.5%, where the active-input count is Binomial(32, 0.125) and spreads 47% per vector. The conduction channels themselves are zero-tune: the analytic expectation built from the config's own tables matches all four of them within 1.75 sigma, the 50% `.bl_cond` anchor at +0.79 sigma. So the -4.08% on the 50% array pin is about five ensemble sigmas — it is the C_WL saturation deficit, not a draw.

**The RS-CSA pair is nearly infeasible against its two constraints.** Writing `E = E_fixed + b * S` with `S` the per-conversion residue integral and `b = mirror_scale * v_rail * t_phase`, the model's `S` spans 38.5 uA (code 0001) to 105.0 uA (code 1111) over the deterministic mid-bin sweep while the ensemble workload means are 16.36 uA (87.5%) and 50.64 uA (50%). Flatness across the two workload points caps `b` from above and the code spread bounds it from below, leaving `b` in [2.120, 2.742] — a window only about 29% wider than the spread floor demands. Both ends bind twice over: `b_lo` on the 1.3 spread floor AND on `E`(87.5%) >= 423 fJ, `b_hi` on holding both points inside +-10%; the 1.8 spread ceiling never binds. `b = 2.3136` pins the spread at 1.3 x 1.015 = 1.3195, a 1.5% deterministic margin on that floor (the code sweep is sample-free), and spends the rest of the freedom on symmetric flatness.

## Hard gates — 5 / 5 PASS

| Gate | Result |
|---|---|
| golden transfer + asymmetric-value regression | 4096 / 4096 codes equal the closed-form transfer built from the config's own tables (0 tap-boundary samples excluded); the deterministic value sweep 0..7 reads codes [0, 2, 4, 6, 9, 11, 13, 15] |
| I_TBL table + 30 nA bound | radix-scaled LRS currents 0.50 / 1.00 / 2.00 uA vs the measured 0.50 / 1.00 / 1.99 uA; worst-plane HRS 30.0 nA at the 30 nA bound |
| RS-CSA 470 fJ flat + code spread (calibration consistency) | per-conversion 429.1 fJ (87.5%) and 508.1 fJ (50%) vs the 470 fJ anchor; per-code spread 1.319x inside [1.3, 1.8] |
| zero input -> code 0 | 64 / 64 outputs read code 0 — the seated PH0 cancels the row leakage floor exactly |
| derived T_AC = 66 ns | PH0 + PH1 + PH2 + PH3 + t4 = 66.0 ns, identical to the modelled per-access latency |

The golden-transfer gate is the load-bearing one: it validates the whole functional path — encode LUT, plane fold, plane-major radix place values, the floor-driven redundant plane, the seated PH0, and the uniform quantizer — against a closed form derived only from the config's own numbers. Its asymmetric-value half pins the LSB-first digit order specifically: each value 0..7 is stored on one column and read under an all-ones input, so a reversed (MSB-first) digit order lands the same value on different T2 slice multipliers and the codes stop matching. The I_TBL table gate is therefore transitively a model gate, not just a config check. The 30 nA bound admits two readings and the gate takes the tighter one — the RAW per-plane current `m * i_t2[drive][HRS]`, which the model saturates exactly; read instead as the plane's excess over the V_X = 0 leakage floor, the model sits at 18.6 nA, 11.4 nA under the bound.

## Energy channels

The model bills five non-zero dynamic rows plus two static seats, the dynamic conduction rows as `E = V * I * t` branch atoms over the derived 66 ns access window. A row's power below is its per-access energy averaged over the DECLARED 66 ns leakage window (`anchors.toml`), the duty period the harness holds apart from the access time; the two seats are leakage powers already. Powers are quoted PER DIE: the ensemble's static leakage is divided by `n_w`, and the modelled latency is the per-die serial access time, the dies being parallel.

| Channel | Owner / rate | 87.5% uW | 50% uW |
|---|---|--:|--:|
| `array` | array + cell, PER ACCESS: WL wire + WL gate caps, selected-cell X dip | 1.092 | 1.097 |
| `.bl_cond` | macro, PER ACCESS: 0.3 V input-branch conduction over T_AC | 13.210 | 52.731 |
| `.bl_cap` | macro, PER VECTOR: BL-column charge (levels held across the row scan) | 0.008 | 0.030 |
| `.dl_cond` | macro, PER ACCESS: 0.8 V row branch (raw I_TBL) over T_AC | 4.045 | 13.752 |
| `rscsa` | RS-CSA, PER CONVERSION: E_fixed + per-phase E_code | 6.501 | 7.698 |
| `mux_driver` | flat seat, STATIC leakage | 5.630 | 5.630 |
| `timing_ctrl` | flat seat, STATIC leakage | 14.030 | 14.030 |
| **TOTAL** | | **44.517** | **94.969** |

Four further rows are structurally zero and are listed, not dropped, so a row that starts drawing energy cannot slip past the pooling unnoticed: `.mux_driver` and `.timing_ctrl` (the seats carry no per-op dynamic share), `bl_driver` and `sl_driver` (ideal sources — the macro bills the whole input branch, and the SL rail is grounded).

## Dual-caliber power report (NOT gated)

A caliber is a MOUNTING HYPOTHESIS about the measured setup: it names the one conduction branch the evaluation instrument feeds, which therefore draws from no macro supply and appears in NO measured power pin.

| Caliber | Mounting | Array pin | RS-CSA pin | OFF-PIN (instrument-fed) |
|---|---|---|---|---|
| X | BL inputs driven off-chip by the board DAC array (Fig.15) | `array` + `.dl_cond` | `rscsa` | `.bl_cond` + `.bl_cap` |
| Y | TBL clamp-driven from outside | `array` + `.bl_cond` + `.bl_cap` | `rscsa` | `.dl_cond` |

Neither branch's rail reaches the converter supply, so under both calibers the RS-CSA pin carries `rscsa` alone — no conduction row is ever pooled into it. The two seats map to their own pins unchanged, and the two structurally-zero driver rows follow their branch (`bl_driver` off-pin under X, `sl_driver` on the array pin under both). Pins plus off-pin are an exhaustive partition of the dynamic rows — the harness asserts it for both calibers at both points — so the quantity each Fig.19 total is compared against is the ON-CHIP TOTAL, the model total minus the off-pin branch.

### 87.5% input sparsity — closes under caliber X, reported not fitted

Full model, every branch billed: 44.517 uW, 2.938 pJ/out (anchor 2.11), EF 21.78 TOPS/W. Draw spread over the 8 rounds: 44.517 +- 0.188 uW (std / sqrt(rounds)).

| Block | caliber Y uW | caliber X uW | anchor uW |
|---|--:|--:|--:|
| Array | 14.310 (2.76x) | 5.138 (0.99x) | 5.178 |
| RS-CSA | 6.501 (0.91x) | 6.501 (0.91x) | 7.127 |
| Mux & Driver | 5.630 (1.00x) | 5.630 (1.00x) | 5.625 |
| Timing & Mode Ctrl | 14.030 (1.00x) | 14.030 (1.00x) | 14.030 |
| off-pin (instrument-fed) | 4.045 | 13.218 | — |
| **ON-CHIP TOTAL** | **40.472 (1.266x)** | **31.299 (0.979x)** | **31.960** |

On-chip energy efficiency: 23.96 TOPS/W under caliber Y, 30.98 TOPS/W under caliber X.

### 50% input sparsity — closes under caliber Y, physics calibration anchor

Full model, every branch billed: 94.969 uW, 6.268 pJ/out (anchor 5.47), EF 10.21 TOPS/W. Draw spread over the 8 rounds: 94.969 +- 0.202 uW (std / sqrt(rounds)).

| Block | caliber Y uW | caliber X uW | anchor uW |
|---|--:|--:|--:|
| Array | 53.858 (0.96x) | 14.849 (0.26x) | 56.150 |
| RS-CSA | 7.698 (1.08x) | 7.698 (1.08x) | 7.133 |
| Mux & Driver | 5.630 (1.00x) | 5.630 (1.00x) | 5.640 |
| Timing & Mode Ctrl | 14.030 (1.00x) | 14.030 (1.00x) | 14.017 |
| off-pin (instrument-fed) | 13.752 | 52.761 | — |
| **ON-CHIP TOTAL** | **81.216 (0.979x)** | **42.207 (0.509x)** | **82.940** |

On-chip energy efficiency: 11.94 TOPS/W under caliber Y, 22.97 TOPS/W under caliber X.

### Finding: each Fig.19 point closes under one mounting, and no mounting closes both

Each measured point is FULLY consistent under exactly ONE mounting hypothesis — all four pins AND the on-chip total at once, every one inside about 10%. At 87.5% that is caliber X: array 0.99x, RS-CSA 0.91x, both seats 1.00x, on-chip total 0.979x. At 50% it is caliber Y: array 0.96x, RS-CSA 1.08x, both seats 1.00x, on-chip total 0.979x. Crossing the mountings closes neither point: caliber Y at 87.5% reads the array pin 2.76x and the on-chip total 1.266x, and caliber X at 50% reads 0.26x and 0.509x. The two hypotheses are mutually exclusive, so no single mounting of the macro accounts for both published points — and neither miss is a tuning shortfall. The 50% array pin carries the C_WL saturation deficit openly: 0.96x here, where a single seed-0 weight matrix read 1.01x. That agreement was a lucky draw, and the ensemble replaces it with the knob-limited number.

The physics behind that exclusion is one-sided. The 87.5% array anchor (5.178 uW) sits BELOW the model's own lower bound on input-branch conduction:

    all-HRS floor at 87.5% = v_bl_in1 * g_cell_on(HRS) * v_bl_in1, over 4 mean active inputs x 3 weight
    planes = 0.3 V * 2.279 uA * 12 = 8.21 uW,

i.e. even a matrix with every cell at HRS — the least current the divider can draw while those inputs are raised, whatever the weights — already exceeds the anchor by 1.6x. So the 87.5% point can never sit on a pin that carries `.bl_cond`, at any weight statistics and under any calibration: caliber Y is ruled out there on physics alone, not on residual size. Physics is calibrated at the 50% point (the caliber-Y array pin); the 87.5% point is reported, never fitted.

Read through caliber X, the 87.5% point draws 31.299 uW on chip = 2.066 pJ per output, i.e. EF 30.98 TOPS/W against the paper's 30.34 TOPS/W headline (1.02x). That agreement IMPLIES the headline excludes input-drive power, which this mounting hands to the instrument; it is a consequence of the reading, not independent support for it. The full model, which bills every branch at full rail for the whole window, reads 21.78 / 10.21 TOPS/W at the two points. Those are the physics view of the macro as a self-contained circuit and are what a system-level estimate should carry.

## Sensitivity

**BL held across the scan vs re-driven.** `.bl_cap` is billed once per input vector, because the BL levels are held across the whole output scan — that is the semantics of one `vec_mat_mul`. A scan is 64 accesses, one per output column, so re-driving the BL column at every access would scale the row by 64 (63 extra charge events per vector): 1.92 uW at the 50% point (+1.89 uW, +3.5% on the caliber-Y array pin) and 0.51 uW at 87.5%. The dual-caliber conclusion is unchanged either way.

**The readout reference current is an operating point the paper does not pin for the measured run.** The `reference_config.i_refs__uA = [[7.0]]` value is `[derived]`: Fig.11(a) annotates I_LSB = 7 uA, and it closes against 224 max MAC units x 0.5 uA per unit = 112 uA full scale over 16 codes. The instrument setting used for the Fig.19 power measurement is unreported. The RS-CSA scales that one current into its whole decision ladder, so it moves both the transfer (which MAC magnitude reads which code) and the residue integrals that set `E_code`; the conduction channels, which dominate the array pin, do not depend on it.

**Weight sparsity is read at VALUE level.** The draw is `P(w = 0) = 0.5`, otherwise uniform over [1, 7] — mean weight 2.0. Drawing each of the three binary planes independently at the same probability would instead give `P(w = 0) = 0.125` and mean weight 3.5, raising the per-active-input BL conductance from `24.0253 * 2 + 7.5975 * 5` to `24.0253 * 3.5 + 7.5975 * 3.5` uS — `.bl_cond` by about 29% and with it the caliber-Y array pin by about 28%. That overshoots the 50% array anchor the model is calibrated at, so the anchor itself discriminates the two readings and selects the value-level one.

## Solver

`n_outer / n_inner = 4 / 3`: the one-active-row step1 solve is light and near-linear. The golden-transfer gate and the scheme unit tests confirm convergence; the run config for a formal sweep is provided under `tools/`.
