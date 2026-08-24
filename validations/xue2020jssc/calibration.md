# xue2020jssc calibration campaign

Device cpu; n_w=4, n_x=32, repeat=4, seed=0; P(x!=0)=0.2935, P(w!=0)=1.0000. The script reports values only.

## 0. Activity calibration input

The declared activity point comes from a prior joint sweep against the two Fig.18 read-path pair targets. The minimum-error point reaches the dense-weight boundary P(w!=0)=1.0 and uses P(x!=0)=0.2935 among at most nine candidate rows (about 2.64 nonzero values per input on average). Conditional nonzero magnitudes remain uniform. This script treats that effective activity point as an input; it does not claim that the paper reports either probability.

## 1. Fixed 1T1R cell input

WL-off g [uS]: (0.001, 0.001).
WL-on g [uS]: (23.8, 138.9).
WL-off V_X ratios: (0.0, 0.0).
WL-on V_X ratios: (0.98, 0.89).

Solver input: n_outer=4, n_inner=3. The separate residual campaign found a 3/3 plateau and added one outer margin.

## 2. ADC reference ladder

I_SUB [uA]: 0.000000, 2.119690, 4.239380, 6.359071, 8.402020, 10.444969, 12.487917, 14.451145
I_REF [uA]: 1.059800, 3.179500, 5.299200, 7.380500, 9.423500, 11.466000, 13.470000
Decode check: PASS ([0, 1, 2, 3, 4, 5, 6, 7]).

## 3. CABLC+DSWCT conduction

Measured rows before fitting [fJ/access]: .control=6553.6011, .cablc=6438.3794, array=0.0000, dswct=2414.0048, sinwp_sc=1564.4978, pn_isub=2090.7656, tmcsa=2981.7875.
Input conduction 8852.3842 fJ/access + declared array capacitance 0.0000 fJ/access; Fig.18 pair target 8464.5000 fJ/access.
Array node caps remain the declared zero paper-reproduction seat; no residual capacitance is fitted.

## 4. PN-ISUB decision energy

Before fitting: SINWP-SC + PN-ISUB 3655.2633 fJ/access; Fig.18 pair target 3655.1250 fJ/access.
PN-ISUB residual 47.665414 fJ per (slot, IO); rounded decision energy 47.665 fJ.

## 5. TMCSA conduction

Pair checks: CABLC+DSWCT 8852.3842/8464.5000 fJ/access; SINWP-SC+PN-ISUB 3655.2633/3655.1250 fJ/access.
Fixed switching 600.0000 fJ/access (3 steps x 4 IO x 50.0 fJ); PH2/PH3 durations remain 0.55980/0.93300 ns; residual conduction_scale 3.2510341, rounded to 3.2510.

## Values to write back

- `reference_config.i_refs__uA = [[1.0598, 3.1795, 5.2992, 7.3805, 9.4235, 11.466, 13.470]]`
- `pn_isub_config.e_per_op__fJ = 47.665`
- `tmcsa_config.e_per_step__fJ = 50.000`
- `tmcsa_config.conduction_scale = 3.2510`
