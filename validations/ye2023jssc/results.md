# ye2023jssc validation results

Outcome of `make validate_ye2023jssc` on CPU (seed 0, N 64, batch 8, float64). Gate targets and reference
targets are read from `anchors.toml`; params from `params.toml`.

The campaign has two parts. Five HARD GATES decide pass/fail — the harness exits non-zero unless all five
pass. The dual-caliber power report is NOT gated: each Fig.19 point is fully consistent with the model under
exactly one mounting hypothesis for the measured setup, and no single mounting explains both, so the two
readings are reported side by side instead of one being fitted.

Four gates are INDEPENDENT — no free parameter is solved against the quantity they check. The RS-CSA energy
gate is a CALIBRATION-CONSISTENCY check: `mirror_scale` and `e_fixed_per_op__fJ` are solved against exactly
its two constraints, so it asserts that solve is jointly feasible and holds across both workloads, not that
the model predicts the 470 fJ anchor.

## Calibration inventory

Exactly five numbers are free. Three carry `[calibrated]` in `params.toml`, two are `[transcribed]` paper
block powers; every other field carries a non-free provenance tag and is pinned by it.

| Parameter | Value | Solved against | Bound | Residual |
|---|--:|---|---|---|
| `c_wl__fF` (the C_WL knob) | 1.504 | the 50% caliber-Y array pin (`array` + `.bl_cond` + `.bl_cap` vs 56.150 uW) | C_WL in [60, 200] fF/row | C_WL = 200.0 fF/row, the UPPER EDGE; array pin +0.60% at seed 0 |
| `mirror_scale` | 0.1452 | jointly: 470 fJ/conversion within +-10% at BOTH sparsity points AND per-code spread in [1.3, 1.8] | b = 16 * mirror_scale in [2.174, 2.724] | b = 2.3232; -8.7% / +9.2% flat, spread 1.320x |
| `e_fixed_per_op__fJ` | 393.3 | (the same constraint pair) | (the same band) | (the same) |
| `mux_driver_config.leakage_per_inst__uW` | 5.63 | Fig.19 Mux & Driver block power, flat across both points | — | 1.00x at both points |
| `timing_ctrl_config.leakage_per_inst__uW` | 14.03 | Fig.19 Timing & Mode Ctrl block power, flat across both points | — | 1.00x at both points |

Fixed BEFORE the C_WL sweep, and not free: `c_bl__fF` = 0.250 and `c_x__fF` = 0.150 as 28nm-class priors.
They put the BL column aggregate at `bl_first_c + 63*bl_segment_c + 64*(c_bl + c_x)` = 29.57 fF, the top of
the 10-30 fF estimate band, and `c_x` inside the 0.05-0.3 fF/cell band. Their energy weight is small
(`.bl_cap` is 0.03 uW at the 50% point), so the C_WL solve is insensitive to where in the bands they sit.

**The C_WL knob is SATURATED.** Closing the 50% array pin exactly would need about 573 fF/row; the sanctioned
band tops out at 200 fF/row, so C_WL sits at the bound edge and the remaining pin residual is reported rather
than zeroed. There is no headroom left: any later drift in `.bl_cond` (a change to `g_cell_on`, to the radix
path, or to the weight-sparsity convention) moves the residual with nothing to absorb it.

**The residual is draw-dependent.** `measure()` programs ONE weight matrix per sparsity point, and the
per-matrix spread of the array pin is about 2.7% (1 sigma) on top of about 2.2% from 64 input vectors. The
seed-0 matrix is about 4.5% denser than the ensemble, so this run reads +0.60% (56.485 vs 56.150 uW) while a
16-draw ensemble of the same configuration reads -3.62% (54.117 uW). Both are inside the 5% acceptance;
quoting either number requires quoting the draw count with it.

**The RS-CSA pair is nearly infeasible against its two constraints.** Writing `E = E_fixed + b * S` with `S`
the per-conversion residue integral and `b = mirror_scale * v_rail * t_phase`, the model's `S` spans
38.5 uA (code 0001) to 105 uA (code 1111) while the workload means are 15.76 uA (87.5%) and 50.26 uA (50%).
Flatness across the two workload points caps `b` from above and the code spread bounds it from below, leaving
`b` in [2.174, 2.724] — a window only about 25% wider than the spread floor demands. `b = 2.3232` keeps a
1.5% deterministic margin on that floor (the code sweep is sample-free) and spends the rest on the
sample-dependent flatness.

## Hard gates — 5 / 5 PASS

| Gate | Result |
|---|---|
| golden transfer + asymmetric-value regression | 512 / 512 codes equal the closed-form transfer built from the config's own tables (0 tap-boundary samples excluded); the deterministic value sweep 0..7 reads codes [0, 2, 4, 6, 9, 11, 13, 15] |
| I_TBL table + 30 nA bound | radix-scaled LRS currents 0.50 / 1.00 / 2.00 uA vs the measured 0.50 / 1.00 / 1.99 uA; worst-plane HRS 30.0 nA at the 30 nA bound |
| RS-CSA 470 fJ flat + code spread (calibration consistency) | per-conversion 428.9 fJ (87.5%) and 513.1 fJ (50%) vs the 470 fJ anchor; per-code spread 1.320x inside [1.3, 1.8] |
| zero input -> code 0 | 64 / 64 outputs read code 0 — the derived PH0 cancels the row leakage floor exactly |
| derived T_AC = 66 ns | PH0 + PH1 + PH2 + PH3 + t4 = 66.0 ns, identical to the profiled per-access latency |

The golden-transfer gate is the load-bearing one: it validates the whole functional path — encode LUT, plane
fold, plane-major radix place values, the input-0 redundant plane, the derived PH0, and the uniform quantizer
— against a closed form derived only from the config's own numbers. Its asymmetric-value half pins the
LSB-first digit order specifically: each value 0..7 is stored on one column and read under an all-ones input,
so a reversed (MSB-first) digit order lands the same value on different T2 slice multipliers and the codes
stop matching. The I_TBL table gate is therefore transitively a model gate, not just a config check. The
30 nA bound admits two readings and the gate takes the tighter one — the RAW per-plane current
`m * i_t2[IN=1][HRS]`, which the model saturates exactly; read instead as the plane's excess over the input-0
leakage floor, the model sits at 18.6 nA, 11.4 nA under the bound.

## Energy channels

The model bills five non-zero dynamic rows plus two static seats, all in `E = V * I * t` branch atoms over
the derived 66 ns access window.

| Channel | Owner / rate | 87.5% uW | 50% uW |
|---|---|--:|--:|
| `array` | array + cell, PER ACCESS: WL wire + WL gate caps, selected-cell X dip | 1.092 | 1.097 |
| `.bl_cond` | macro, PER ACCESS: 0.3 V input-branch conduction over T_AC | 12.871 | 55.356 |
| `.bl_cap` | macro, PER VECTOR: BL-column charge (levels held across the row scan) | 0.007 | 0.032 |
| `.dl_cond` | macro, PER ACCESS: 0.8 V row branch (raw I_TBL) over T_AC | 3.932 | 14.318 |
| `rscsa` | RS-CSA, PER CONVERSION: E_fixed + per-phase E_code | 6.499 | 7.775 |
| `mux_driver` | flat seat, STATIC leakage | 5.630 | 5.630 |
| `timing_ctrl` | flat seat, STATIC leakage | 14.030 | 14.030 |
| **TOTAL** | | **44.061** | **98.237** |

Four further rows are structurally zero and are listed, not dropped, so a row that starts drawing energy
cannot slip past the pooling unnoticed: `.mux_driver` and `.timing_ctrl` (the seats carry no per-op dynamic
share), `bl_driver` and `sl_driver` (ideal sources — the macro bills the whole input branch, and the SL rail
is grounded).

## Dual-caliber power report (NOT gated)

A caliber is a MOUNTING HYPOTHESIS about the measured setup: it names the one conduction branch the
evaluation instrument feeds, which therefore draws from no macro supply and appears in NO measured power pin.

| Caliber | Mounting | Array pin | RS-CSA pin | OFF-PIN (instrument-fed) |
|---|---|---|---|---|
| X | BL inputs driven off-chip by the board DAC array (Fig.15) | `array` + `.dl_cond` | `rscsa` | `.bl_cond` + `.bl_cap` |
| Y | TBL clamp-driven from outside | `array` + `.bl_cond` + `.bl_cap` | `rscsa` | `.dl_cond` |

Neither branch's rail reaches the converter supply, so under both calibers the RS-CSA pin carries `rscsa`
alone — no conduction row is ever pooled into it. The two seats map to their own pins unchanged, and the two
structurally-zero driver rows follow their branch (`bl_driver` off-pin under X, `sl_driver` on the array pin
under both). Pins plus off-pin are an exhaustive partition of the dynamic rows — the harness asserts it for
both calibers at both points — so the quantity each Fig.19 total is compared against is the ON-CHIP TOTAL,
the model total minus the off-pin branch.

### 87.5% input sparsity — closes under caliber X, reported not fitted

Full model, every branch billed: 44.061 uW, 2.908 pJ/out (anchor 2.11), EF 22.01 TOPS/W.

| Block | caliber Y uW | caliber X uW | anchor uW |
|---|--:|--:|--:|
| Array | 13.970 (2.70x) | 5.025 (0.97x) | 5.178 |
| RS-CSA | 6.499 (0.91x) | 6.499 (0.91x) | 7.127 |
| Mux & Driver | 5.630 (1.00x) | 5.630 (1.00x) | 5.625 |
| Timing & Mode Ctrl | 14.030 (1.00x) | 14.030 (1.00x) | 14.030 |
| off-pin (instrument-fed) | 3.932 | 12.878 | — |
| **ON-CHIP TOTAL** | **40.129 (1.256x)** | **31.183 (0.976x)** | **31.960** |

On-chip energy efficiency: 24.16 TOPS/W under caliber Y, 31.10 TOPS/W under caliber X.

### 50% input sparsity — closes under caliber Y, physics calibration anchor

Full model, every branch billed: 98.237 uW, 6.484 pJ/out (anchor 5.47), EF 9.87 TOPS/W.

| Block | caliber Y uW | caliber X uW | anchor uW |
|---|--:|--:|--:|
| Array | 56.485 (1.01x) | 15.415 (0.27x) | 56.150 |
| RS-CSA | 7.775 (1.09x) | 7.775 (1.09x) | 7.133 |
| Mux & Driver | 5.630 (1.00x) | 5.630 (1.00x) | 5.640 |
| Timing & Mode Ctrl | 14.030 (1.00x) | 14.030 (1.00x) | 14.017 |
| off-pin (instrument-fed) | 14.318 | 55.388 | — |
| **ON-CHIP TOTAL** | **83.920 (1.012x)** | **42.849 (0.517x)** | **82.940** |

On-chip energy efficiency: 11.56 TOPS/W under caliber Y, 22.63 TOPS/W under caliber X.

### Finding: each Fig.19 point closes under one mounting, and no mounting closes both

Each measured point is FULLY consistent under exactly ONE mounting hypothesis — all four pins AND the on-chip
total at once, every one inside about 10%. At 87.5% that is caliber X: array 0.97x, RS-CSA 0.91x, both seats
1.00x, on-chip total 0.976x. At 50% it is caliber Y: array 1.01x, RS-CSA 1.09x, both seats 1.00x, on-chip
total 1.012x. Crossing the mountings closes neither point: caliber Y at 87.5% reads the array pin 2.70x and
the on-chip total 1.256x, and caliber X at 50% reads 0.27x and 0.517x. The two hypotheses are mutually
exclusive, so no single mounting of the macro accounts for both published points — and neither miss is a
tuning shortfall.

The physics behind that exclusion is one-sided. The 87.5% array anchor (5.178 uW) sits BELOW the model's own
lower bound on input-branch conduction:

    all-HRS floor at 87.5% = v_bl_in1 * g_cell_on(HRS) * v_bl_in1, over 4 mean active inputs x 3 weight
    planes = 0.3 V * 2.279 uA * 12 = 8.21 uW,

i.e. even a matrix with every cell at HRS — the least current the divider can draw while those inputs are
raised, whatever the weights — already exceeds the anchor by 1.6x. So the 87.5% point can never sit on a pin
that carries `.bl_cond`, at any weight statistics and under any calibration: caliber Y is ruled out there on
physics alone, not on residual size. Physics is calibrated at the 50% point (the caliber-Y array pin); the
87.5% point is reported, never fitted.

Read through caliber X, the 87.5% point draws 31.183 uW on chip = 2.058 pJ per output, i.e. EF 31.10 TOPS/W
against the paper's 30.34 TOPS/W headline (1.02x). That agreement IMPLIES the headline excludes input-drive
power, which this mounting hands to the instrument; it is a consequence of the reading, not independent
support for it. The full model, which bills every branch at full rail for the whole window, reads 22.01 /
9.87 TOPS/W at the two points. Those are the physics view of the macro as a self-contained circuit and are
what a system-level estimate should carry.

## Sensitivity

**BL held across the scan vs re-driven.** `.bl_cap` is billed once per input vector, because the BL levels are
held across the whole output scan — that is the semantics of one `vec_mat_mul`. If the BL column were instead
re-driven for each of the 64 accesses in a scan, the row would scale by 64: 2.05 uW at the 50% point
(+2.0 uW, +3.6% on the caliber-Y array pin) and 0.45 uW at 87.5%. The dual-caliber conclusion is unchanged
either way.

**`i_lsb__uA` is an operating point the paper does not pin for the measured run.** The 7.0 uA value is
`[derived]`: Fig.11(a) annotates I_LSB = 7 uA, and it closes against 224 max MAC units x 0.5 uA per unit =
112 uA full scale over 16 codes. The instrument setting used for the Fig.19 power measurement is unreported.
I_LSB scales the whole code ladder, so it moves both the transfer (which MAC magnitude reads which code) and
the RS-CSA residue integrals that set `E_code`; the conduction channels, which dominate the array pin, do not
depend on it.

**Weight sparsity is read at VALUE level.** The draw is `P(w = 0) = 0.5`, otherwise uniform over [1, 7] —
mean weight 2.0. Drawing each of the three binary planes independently at the same probability would instead
give `P(w = 0) = 0.125` and mean weight 3.5, raising the per-active-input BL conductance from
`24.0253 * 2 + 7.5975 * 5` to `24.0253 * 3.5 + 7.5975 * 3.5` uS — `.bl_cond` by about 29% and with it the
caliber-Y array pin by about 28%. That overshoots the 50% array anchor the model is calibrated at, so the
anchor itself discriminates the two readings and selects the value-level one.

## Solver

`n_outer / n_inner = 4 / 3`: the one-active-row step1 solve is light and near-linear. The golden-transfer
gate and the scheme unit tests confirm convergence; the run config for a formal sweep is provided under
`tools/`.
