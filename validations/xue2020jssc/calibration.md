# xue2020jssc calibration campaign

Energy-basis calibration of the geometry-dependent DECLARED values in `params.toml` against the `anchors.toml` target. Device cuda, float32, `.eval()`, all-off policy. N = 2000 draws/point, seed 0. Non-circular: control + reference ADOPTED from Fig.18, the read path is pure physics (array IR-drop solve), and p_zero is LOCKED to the read-path share (never solved against the total).

## Step a -- reference ladder re-derivation at this geometry

Unit i_sub staircase I(MAC=0..7) [uA] over 9 live rows:

    0.0000, 1.6171, 3.2341, 4.8512, 6.4539, 8.0566, 9.6593, 11.2445

Adjacent-midpoint reference taps I_REF[0..6] [uA]:

    0.808528, 2.425583, 4.042638, 5.652523, 7.255236, 8.857949, 10.451908

Staircase monotonic: True (driven through the full array IR-drop solve). Ladder verification (code == MAC magnitude): True (codes [0, 1, 2, 3, 4, 5, 6, 7]).

## Step b -- adopted peripheral seats (Fig.18, declared not fitted)

per-sub-array budget = 5.13 mW / 8 = 641.2500 uW (= 32.0600 pJ/access at 20.0 MHz).
- Control 29.2% -> 187.2450 uW = 9362.2500 fJ/access; pure per-op caliber (100% dynamic, 0% static): `e_control_per_op__fJ = 9362.2500`, `control_config.leakage_per_inst__uW = 0.0`.
- Reference 23.7% -> 100% static: `reference_config.leakage_per_inst__uW = 151.97625` uW.

Read-path seats stay pure physics: cablc / sl / adc / pn_isub static seats declared small/zero (NOT solved); g_map, V_BL_CLAMP, wire R, and the conduction windows stay declared structural constants; the capacitive / per-op remainders (array caps, SINWP-SC `c_hold`, TMCSA phase widths + `e_fixed_per_op`) are the pair-caliber energy campaign's domain (their fitted values live in `params.toml` under each block's `[calibrated]` comment; the measured per-slice breakdown is in `results.md`), never reverse-solved to fill the total.

## Step c -- p_zero LOCK to the read-path physics share (15.10 pJ/access)

Read-path target = 47.1% x 32.06 = 15.100 pJ/access.

| p_zero | read-path [pJ/acc] | vs 15.10 |
|--:|--:|--:|
| 0.00 |   22.760 | 1.507x |
| 0.10 |   20.567 | 1.362x |
| 0.20 |   18.389 | 1.218x |
| 0.30 |   16.242 | 1.076x |
| 0.40 |   14.089 | 0.933x |
| 0.50 |   11.912 | 0.789x |
| 0.60 |    9.602 | 0.636x |
| 0.70 |    7.415 | 0.491x |
| 0.80 |    5.161 | 0.342x |
| 0.90 |    2.904 | 0.192x |

**LOCKED p_zero = 0.3530** (marginal P(x=0) ~= 0.515): the p_zero at which the pure-physics read path conducts its Fig.18 47.1% share. Locked to the READ PATH, not the total.

## Step d -- p_zero lock outcome and write-back values

At the LOCKED p_zero = 0.3530, the read-path physics reaches its 15.10 pJ/access
Fig.18 share (step c); the two adopted seats (control, reference) sit at their
own Fig.18 shares by construction (step b). This closes the cell chord (step a's
ladder basis), the ladder re-derivation (step a), and the p_zero lock (step c) --
the record this report carries.

The per-block energy constants (array/cell caps, SINWP-SC `c_hold`, PN-ISUB
`e_per_op`, TMCSA phase widths + `e_fixed_per_op`) belong to the pair-caliber
energy campaign, not to this report: their fitted values live in `params.toml`
under each block's `[calibrated]` provenance comment, and the measured per-slice
breakdown is reported in `results.md`.

Values to write back into params.toml / anchors.toml (print only; nothing is mutated):

- `reference_config.i_refs__uA = [[0.808528, 2.425583, 4.042638, 5.652523, 7.255236, 8.857949, 10.451908]]`  # calibrated ladder
- `e_control_per_op__fJ = 9362.2500`  # adopted (control 100% dynamic, pure per-op)
- `control_config.leakage_per_inst__uW = 0.0`  # adopted (control 0% static)
- `reference_config.leakage_per_inst__uW = 151.97625`  # adopted (reference 100% static)
- `anchors.toml [data].p_zero = 0.3530`  # LOCKED to the read-path share (15.10 pJ), NOT the total

