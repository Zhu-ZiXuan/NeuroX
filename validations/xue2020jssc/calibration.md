# xue2020jssc calibration campaign

> **PENDING RECALIBRATION.** The stage-2 capacitances and the stage-3 constants below were solved against the replaced capacitive form (full-cycle `C*V^2`, WL wire billed by a scheme-local per-plane special case). The macro now composes the kernel `XbarArray1t1r` in `WL_IN_BL_SCAN`, which bills `E = V_rail * C * |dv|` per solve on the two declared rails and covers the WL wire itself, so every cap-bearing seat here needs re-deriving. Stage 1 (currents / the ADC ladder) is unaffected: the flattening is bit-exact.

Energy-basis re-derivation of the geometry-dependent `[calibrated]` values of `params.toml` against the `anchors.toml` target, in the fixed stage order CURRENTS -> CAPACITANCES -> CONSTANTS. Device cuda:1, float32, `.eval()`, all-off policy, solve chunk 4096. Stage measurements: n_w = 32 weight draws x n_x = 256 inputs x 4 rounds, seed 0, at the declared workload p_zero = 0.353; the stage-2 pair solve pools 4 rounds split into 4 statistically independent blocks x 1 rounds, whose spread MEASURES the ill-conditioned cap seat. Non-circular: control + reference are ADOPTED from Fig.18, the read path is pure physics (array IR-drop solve), and p_zero is LOCKED to the read-path share, never solved against the total.

Per-access basis: one access = one MUX-slot conversion set, so a run of n_w x n_x x rounds input draws reads `accesses = n_w * n_x * mux_factor` (32) output accesses and EVERY profiled energy total is divided by that leading count. A per-op seat is divided further by its own event count per access: 16 SINWP-SC hold-cap events (x_bits x IO x P/N) and 12 TMCSA step charges (IO x steps). Each stage proves its own normalization by an n_w / n_x doubling check.

## Stage 1 -- currents

Operating point of the linearized cell chord in force (extracted by `neurox.tools.calibrate_cell` from `tools/params_detail.toml` at V_BL = V_BLC = 0.29 V, V_SL = 0 V, WL off/on 0.0 / 0.9 V): g_cell_on = [5.002494516043828, 94.57440211855133] uS, g_cell_off = [0.00010173896263385642, 0.00010276662892308729] uS, vx_ratio_on = [0.9970223060166784, 0.9428045968043393]. The absolute current scale of every read-path row rides on it, so it is the first thing this campaign fixes.

Unit i_sub staircase I(MAC=0..7) [uA] over 9 selected inputs:

    0.0000, 1.6171, 3.2341, 4.8512, 6.4539, 8.0566, 9.6593, 11.2445

Adjacent-midpoint reference taps I_REF[0..6] [uA]:

    0.808528, 2.425583, 4.042638, 5.652523, 7.255236, 8.857949, 10.451908

Staircase monotonic: True (driven through the full array IR-drop solve). Ladder verification (code == MAC magnitude): True (codes [0, 1, 2, 3, 4, 5, 6, 7]).

Leading-dimension invariance of the probe: re-driving the same staircase tiled over an extra leading axis (x2) reproduces every step to 0.00e+00 % -- the staircase is a PER-CONVERSION current, unaffected by how many leading positions ride one solve, so the ladder needs no leading-dim normalization.

## Stage 2 -- capacitances

Conduction is FROZEN by stage 1; the two paired-slice residuals seat the capacitive remainders. Measured at the incoming values (cap scale x0.71949 on the `params_detail.toml` structure, uniform across all 10 cap nodes to 2.22e-14 %; c_hold = 14.3614 fF), pooled over 4 rounds in 4 independent blocks (pooled round-total relative std 0.000 %):

| Pair | Fig.18 target [pJ/acc] | conduction (frozen) | capacitive term | measured pair | seat solved |
|---|--:|--:|--:|--:|:--|
| cablc+dswct |   8.4638 |   8.2124 |   0.2601 (array row) |   8.4725 | uniform cap scale |
| sinwp_sc+pn_isub |   3.6548 |   3.4248 |   0.2298 (hold caps) |   3.6546 | `c_hold__fF` |

The array row bills capacitance ONLY, so it is exactly proportional to the uniform cap scale: the pair-1 residual 0.2514 pJ/access asks for x0.96681 on the incoming row, i.e. a cumulative **cap scale x0.69561** on the declared `params_detail.toml` structure. The SINWP-SC hold caps cycle 16 times per access, so the pair-2 residual 0.2300 pJ/access seats **c_hold = 14.3768 fF** (the kept 5.0 fJ comparator per-op stays inside the measured PN-ISUB row).

CONDITIONING of the two seats, measured not asserted. The pair-1 residual is only 3.0% of its pair, so the cap-scale seat AMPLIFIES a relative conduction error by conduction/residual = x32.7 -- it is the ill-conditioned seat of this campaign, so its tolerance is MEASURED here rather than asserted. The 4 blocks below are statistically independent (1 rounds each, disjoint seeds), so their spread IS the seat uncertainty:

| Independent block | conduction cablc+dswct | array cap row | solved cap scale | solved c_hold [fF] |
|---|--:|--:|--:|--:|
| block 0 (seeds 0..0) |   8.1529 |  0.26004 | 0.86035 | 16.2518 |
| block 1 (seeds 1..1) |   8.2640 |  0.26008 | 0.55281 | 12.9716 |
| block 2 (seeds 2..2) |   8.1627 |  0.26007 | 0.83306 | 15.5747 |
| block 3 (seeds 3..3) |   8.2700 |  0.26010 | 0.53626 | 12.7093 |
| **pooled (4 rounds)** | **  8.2124** | ** 0.26007** | **0.69561** | **14.3768** |

Block-basis relative 1 sigma: cap scale 25.15 %, c_hold 12.51 % (at 1 rounds). Averaging the 4 blocks divides that by sqrt 4 = 2.0, so the ADOPTED seats carry **cap scale +-12.57 % (1 sigma)** and **c_hold +-6.26 % (1 sigma)** at the 4-round basis. c_hold is well conditioned (its cap term is a large fraction of its pair); the cap scale is not, and its stated tolerance is part of the result. That tolerance is ACCEPTED at this standard basis, not bought down with extra draws: the array cap row is 0.8% of the measured total, so the seat's 1 sigma moves the headline by only +-0.10 %.

Re-measured at the solved values: cablc+dswct = 8.4638 pJ/access = 1.0000x its Fig.18 share, sinwp_sc+pn_isub = 3.6548 pJ/access = 1.0000x.

Leading-dimension invariance (same macro, one round per point, both draw dimensions doubled in turn):

| Row [pJ/access] | baseline | n_w x2 | n_x x2 | max dev |
|---|--:|--:|--:|--:|
| `.control` |   9.36225 |   9.36225 |   9.36225 | 0.000 % |
| `.cablc` |   5.93232 |   5.92716 |   5.93720 | 0.087 % |
| `array` |   0.25140 |   0.25143 |   0.25142 | 0.008 % |
| `dswct` |   2.22058 |   2.21892 |   2.22248 | 0.086 % |
| `sinwp_sc` |   1.69768 |   1.69893 |   1.70050 | 0.166 % |
| `pn_isub` |   1.92716 |   1.92757 |   1.92934 | 0.113 % |
| `tmcsa` |   2.97328 |   2.97268 |   2.97297 | 0.020 % |
| **total** |  31.96348 |  31.95775 |  31.97498 | 0.036 % |
| accesses (leading count) | 262144 | 524288 | 524288 | x2.0 / x2.0 |
| raw profiled total [nJ] |  8379.036 | 16755.067 | 16764.100 | (scales with the leading count) |

| Re-solved from that point | cap scale | c_hold [fF] |
|---|--:|--:|
| baseline | 0.86035 | 16.2518 |
| n_w x2 | 0.87913 | 16.1481 |
| n_x x2 | 0.84150 | 15.9392 |

That re-solve table proves the NORMALIZATION, not the seat: each point is a SINGLE round, so its cap-scale column carries the full x32.7 amplification of one round's conduction noise and scatters far wider than the 4-round seat adopted above. The rows that must hold flat are the per-access rows in the preceding table -- and they do.

## Stage 3 -- constants

The TMCSA slice carries two unknowns against one constraint, so `e_fixed_per_op__fJ` is PINNED at its declared plausibility ceiling 150.0 fJ per conversion step (latch + coupling caps + the folded-in PH1 bias at 55 nm) and the single phase-window scale carries the residual. Measured at the as-drawn Fig.10(b) occupancy (PH2 18% + PH3 30% of each step, scale x1):

| Term | [pJ/access] |
|---|--:|
| Fig.18 target (9.3 %) |   2.9816 |
| per-step constant (12 charges x 150.0 fJ) |   1.8000 |
| PH2/PH3 branch conduction at the as-drawn widths |   0.7653 |
| conduction budget left by the constant |   1.1816 |

**Phase-window scale x1.54401** on the as-drawn widths: t_ph2 = [0.878232, 0.853219, 0.864336] ns, t_ph3 = [1.463719, 1.422031, 1.440559] ns, so PH2+PH3 occupy 74.1% of each conversion step and PH1/PH4 the rest. TENSION, reported not hidden: the as-drawn occupancy is 48%, so the slice cannot close with BOTH the as-drawn widths and the e_fixed ceiling holding; the widening is the residual compromise.

Re-measured at the solved windows: tmcsa = 2.9816 pJ/access = 1.0000x its Fig.18 share.

ADOPTED peripheral seats (declared from the Fig.18 shares, never fitted): per-sub-array budget = 5.13 mW / 8 = 641.2500 uW = 32.0600 pJ/access at 20.0 MHz. Control 29.2 % -> 187.2450 uW = 9362.2500 fJ/access, billed as a PURE per-op constant (100 % dynamic, `control_config.leakage_per_inst__uW = 0.0000`). Reference 23.7 % -> 100 % static: `reference_config.leakage_per_inst__uW = 151.97625` uW. Measured control row: 9.3622 pJ/access.

Leading-dimension invariance (same macro, one round per point, both draw dimensions doubled in turn):

| Row [pJ/access] | baseline | n_w x2 | n_x x2 | max dev |
|---|--:|--:|--:|--:|
| `.control` |   9.36225 |   9.36225 |   9.36225 | 0.000 % |
| `.cablc` |   5.93232 |   5.92716 |   5.93720 | 0.087 % |
| `array` |   0.25140 |   0.25143 |   0.25142 | 0.008 % |
| `dswct` |   2.22058 |   2.21892 |   2.22248 | 0.086 % |
| `sinwp_sc` |   1.69768 |   1.69893 |   1.70050 | 0.166 % |
| `pn_isub` |   1.92716 |   1.92757 |   1.92934 | 0.113 % |
| `tmcsa` |   2.97324 |   2.97264 |   2.97293 | 0.020 % |
| **total** |  31.96344 |  31.95771 |  31.97494 | 0.036 % |
| accesses (leading count) | 262144 | 524288 | 524288 | x2.0 / x2.0 |
| raw profiled total [nJ] |  8379.025 | 16755.046 | 16764.079 | (scales with the leading count) |

| Re-solved from that point | phase-window scale | e_control_per_op [fJ] |
|---|--:|--:|
| baseline | 1.55498 | 9362.2500 |
| n_w x2 | 1.55577 | 9362.2500 |
| n_x x2 | 1.55540 | 9362.2500 |

## p_zero LOCK to the read-path physics share

Read-path target = 47.1 % x 32.06 = 15.100 pJ/access, scanned at one round per point on the calibrated macro.

| p_zero | read-path [pJ/acc] | vs 15.10 |
|--:|--:|--:|
| 0.00 |   21.392 | 1.417x |
| 0.10 |   19.540 | 1.294x |
| 0.20 |   17.765 | 1.176x |
| 0.30 |   15.987 | 1.059x |
| 0.40 |   14.160 | 0.938x |
| 0.50 |   12.327 | 0.816x |
| 0.60 |   10.498 | 0.695x |
| 0.70 |    8.591 | 0.569x |
| 0.80 |    6.610 | 0.438x |
| 0.90 |    4.637 | 0.307x |

**LOCKED p_zero = 0.3485** (marginal P(x=0) ~= 0.511): the p_zero at which the pure-physics read path conducts its Fig.18 47.1 % share. Locked to the READ PATH, not the total; the declared `anchors.toml` value is 0.353.

## Total + breakdown at the declared p_zero

At the declared p_zero = 0.3530: total = 32.061 pJ/access = 1.000x (err +0.0%), within +-5%: PASS (the total FALLS OUT of the read-path seats + the adopted seats; it is never solved for).

| Slice | Energy [pJ/acc] | dyn | static | Fig.18 x 32.06 [pJ] | pred/ref | basis |
|---|--:|--:|--:|--:|--:|:--|
| control |    9.362 |   9.362 |  0.000 |    9.362 |  1.00x | adopted |
| reference |    7.599 |   0.000 |  7.599 |    7.598 |  1.00x | adopted |
| cablc+dswct |    8.464 |   8.464 |  0.000 |    8.464 |  1.00x | physics pair |
| sinwp_sc+pn_isub |    3.655 |   3.655 |  0.000 |    3.655 |  1.00x | physics pair |
| tmcsa |    2.982 |   2.982 |  0.000 |    2.982 |  1.00x | physics |
| cablc |    6.224 |   6.224 |  0.000 |     -    |   -    | pair member |
| dswct |    2.240 |   2.240 |  0.000 |     -    |   -    | pair member |
| sinwp_sc |    1.710 |   1.710 |  0.000 |     -    |   -    | pair member |
| pn_isub |    1.944 |   1.944 |  0.000 |     -    |   -    | pair member |
| **TOTAL (gated)** | **  32.061** |  24.463 |  7.599 | **  32.060** | **1.000x** | PASS (+-5%, err +0.0%) |

## Values to write back into params.toml (print only; nothing is mutated)

- `reference_config.i_refs__uA = [[0.808528, 2.425583, 4.042638, 5.652523, 7.255236, 8.857949, 10.451908]]`  # stage 1, calibrated ladder
- `array_config.bl_first_c__fF = 0.1391216`  # stage 2, cap scale x0.69561
- `array_config.bl_segment_c__fF = 0.0055648638`  # stage 2, cap scale x0.69561
- `array_config.sl_first_c__fF = 0.1391216`  # stage 2, cap scale x0.69561
- `array_config.sl_segment_c__fF = 0.0055648638`  # stage 2, cap scale x0.69561
- `array_config.wl_first_c__fF = 0.1391216`  # stage 2, cap scale x0.69561
- `array_config.wl_segment_c__fF = 0.0055648638`  # stage 2, cap scale x0.69561
- `array_config.cell_config.c_bl__fF = 0.1391216`  # stage 2, cap scale x0.69561
- `array_config.cell_config.c_x__fF = 0.20868239`  # stage 2, cap scale x0.69561
- `array_config.cell_config.c_sl__fF = 0.069560798`  # stage 2, cap scale x0.69561
- `array_config.cell_config.c_wl__fF = 0.1391216`  # stage 2, cap scale x0.69561
- `sinwp_sc_config.c_hold__fF = 14.3768`  # stage 2, pair-2 residual over 16 events/access
- `tmcsa_config.t_ph2_per_step__ns = [0.878232, 0.853219, 0.864336]`  # stage 3, as-drawn x1.54401
- `tmcsa_config.t_ph3_per_step__ns = [1.463719, 1.422031, 1.440559]`  # stage 3, as-drawn x1.54401
- `tmcsa_config.e_fixed_per_op__fJ = 150.0`  # stage 3, pinned plausibility ceiling
- `e_control_per_op__fJ = 9362.2500`  # stage 3, adopted (control 100 % dynamic, pure per-op)
- `control_config.leakage_per_inst__uW = 0.0000`  # stage 3, adopted (control 0 % static)
- `reference_config.leakage_per_inst__uW = 151.97625`  # stage 3, adopted (reference 100 % static)

Read-path physical declarations (g_map, V_BL_CLAMP, wire R, conduction windows) are left as declared, the read-path static seats stay declared small/zero, and `anchors.toml` is not a write-back target of this campaign: the p_zero lock above is reported against its declared value, never written.

