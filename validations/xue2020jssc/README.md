# xue2020jssc validation

This directory describes one 256×512 1T1R sub-array from Xue et al. (JSSC 2020). `validate.py` compares its modeled energy with the 32.0625 pJ/access target derived in `anchors.toml` from the paper's simulated 5.13 mW whole-macro result. `tools/calibrate.py` takes the fixed cell and CABLC models plus the declared activity distribution as inputs, derives the geometry-dependent reference ladder, and reports the PN-ISUB decision energy and TMCSA conduction scale required by the Fig.18 targets without modifying `params.toml`.

## Run

    make validate_xue2020jssc

The workload convention comes from `anchors.toml`. Each input first samples at most nine candidate row positions uniformly without replacement; each candidate is nonzero with calibrated probability 0.2935, and its conditional nonzero value is uniform over 1–3. Weights are uniform over the six nonzero sign-magnitude values ±1–±3; the calibrated nonzero-probability boundary is 1.0. These are effective activity parameters inferred from the two Fig.18 read-path pair targets, not measured workload statistics reported by the paper. Device, draw count, random seed, and solve chunk are run controls accepted by `validate.py`.

## Files

- `params.toml`: paper design and current model parameters.
- `policy.toml`: all-off nonideality policy.
- `anchors.toml`: paper energy target, Fig.18 shares, and workload conventions.
- `validate.py`: profiler-driven validation campaign.
- `tools/calibrate.py`: ADC-ladder and read-path-energy derivation; it writes a report but never edits parameters.
- `results.md`: validation result for the checked-in parameters.
- `calibration.md`: commands, workload, fitted values, and solver result of the current campaign.

## Geometry and current path

The 128 logical columns use two magnitude digits and two polarities, giving 512 physical columns. A 32:1 column MUX leaves four CIM-IO lanes. DSWCT applies the weight-digit mirror ratios while retaining the digit axis; SINWP-SC then applies the input-bit ratios and sums both radix axes. PN-ISUB forms the signed magnitude, and the TMCSA/SAR pair quantizes it.

For the shipped radix-2, two-digit, two-bit operating point, the LSB-first DSWCT ratios are `(0.25, 0.5)` and the SINWP-SC ratios are `(0.25, 0.5)`. Their products reproduce the four place-value coefficients `(1/16, 1/8, 1/8, 1/4)`.

## Timing

The first `K-1` input bits each occupy `t_sample__ns`; the last bit enters the settle-and-sense tail directly. At ADC resolution `b`:

`tail(b) = t_settle__ns + b * latency_per_step__ns`

`access_latency(b) = (K - 1) * t_sample__ns + tail(b)`

The measured 1-bit and 2-bit macro latencies imply 2.85 ns for the extra sampled input phase. The paper's simulated sensing-step latencies are 3.16, 3.07, and 3.11 ns; this model assumes one uniform 3.11 ns decision period and derives a 2.42 ns settle remainder so the modeled three-bit access latency is 14.60 ns. The initiation interval is the 50 ns period of the paper's 20 MHz simulated power point; the remaining time is implicit idle time. A full VMM serializes `mux_factor` accesses, so both circuit latency and initiation interval are multiplied by 32.

Array/CABLC/DSWCT use one diagonal window per input bit: sampled bits use 2.85 ns and the live bit uses the runtime tail. SINWP-SC uses suffix-hold windows. PN-ISUB uses the tail only. These windows are selected from the requested ADC width at runtime.

## Energy ownership

| Fig.18 slice | Model owner |
|---|---|
| Control | 70% macro per-access event; 30% control-block leakage integrated over 50 ns |
| Reference | static current-reference seat |
| CABLC | macro `cablc` channel for the complete VDD·I input branch; array row for node capacitance, declared zero in this paper-reproduction preset |
| DSWCT | per-digit current-mirror conduction |
| SINWP-SC | held/live mirror-leg conduction; no fitted hold-cap term |
| PN-ISUB | three internal current branches plus sign-decision per-op energy |
| TMCSA | PH2/PH3 conduction plus one fixed switching event per sensing step and CIM-IO |

The kernel `SarIadc` is energy-silent in this macro; it supplies values and timing only. TMCSA bills 50 fJ per sensing step and CIM-IO for its internal latch, reset nodes, and local switching, while its PH2/PH3 conduction law carries the calibrated residual scale recorded in `params.toml`. An external DOUT register is not included.

## Calibration stages

The cell conductance table and 100 Ω CABLC output resistance are fixed inputs, not calibration variables. The ADC stage probes the unit `I_SUB` staircase through the complete array solve and places reference taps at adjacent midpoints; it does not use the energy target.

The read-path campaign jointly scans input and weight nonzero probabilities against the CABLC+DSWCT and SINWP-SC+PN-ISUB conduction targets while retaining uniform conditional nonzero magnitudes. The minimum-error point reaches the dense-weight boundary and uses an input-candidate nonzero probability of 0.2935. At that activity point, the declared array-node capacitance seat is zero because explicit CABLC+DSWCT conduction already slightly exceeds the paired target; this is a paper-reproduction convention, not a claim that the physical capacitances vanish. The campaign rounds PN-ISUB's fitted three-inverter/latch residual to 47.665 fJ per decision and fits TMCSA `conduction_scale` to the residual after subtracting its declared 50 fJ per-step switching event, while keeping the assumed PH2/PH3 durations fixed.

Control and reference remain adopted Fig.18 seats. Because the activity probabilities and two remaining read-path parameters are inferred from the same breakdown later used for comparison, agreement is a model-consistency result rather than independent validation.
