# xue2020jssc validation

This directory describes one 256×512 1T1R sub-array from Xue et al. (JSSC 2020). `validate.py` compares its modeled energy with the 32.0625 pJ/access target derived in `anchors.toml` from the paper's simulated 5.13 mW whole-macro result. The standard solver tool derives numerical iteration counts, the standard ADC probe reports the nominal input clusters used to choose references manually, and the validation command writes the energy comparison to `validation.log` and `energy_breakdown.svg` under its output directory.

## Run

    make validate_xue2020jssc

The default output directory is `log/validation/xue2020jssc`; pass `--output-dir` when invoking `validate.py` directly to keep another run separately. The breakdown SVG compares NeuroX with Fig.18 in pJ/access using the campaign's paired accounting basis.

The workload convention comes from `anchors.toml`. Each input first samples at most nine candidate row positions uniformly without replacement; each candidate is nonzero with calibrated probability 0.28, and its conditional nonzero value is uniform over 1–3. Each weight is nonzero with calibrated probability 0.8; conditional nonzero weights are uniform over the six sign-magnitude values ±1–±3. These are effective activity parameters inferred from the two Fig.18 read-path pair targets after adopting 0.5 Ω wire segments, not measured workload statistics reported by the paper. Device, draw count, random seed, and solve chunk are run controls accepted by `validate.py`. The ADC characterization is a separate nominal-range campaign and places its nine candidate positions in a contiguous row window, matching the mapped 3x3-kernel block.

## Files

- `params.toml`: paper design and current model parameters.
- `policy.toml`: all-off nonideality policy.
- `anchors.toml`: paper energy target, Fig.18 shares, and workload conventions.
- `validate.py`: profiler-driven validation campaign.
- `tools/calibrate_solver.toml`: standard array-solver iteration sweep.
- `tools/calibrate_adc.toml`: nominal ADC-input campaign for the fixed macro mode and contiguous nine-row block; one run stores random observations separately, then conditionally supplements every under-covered signed ideal value.
- `tools/analyze_adc_margin.py`: self-contained Xue2020 magnitude folding, fixed-boundary margin analysis, and plot entry point over saved probe data.
- `tools/_adc_plot.py`: the magnitude ridgeline and per-cluster statistics plots used by that script.

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
| Control | 70% control-block per-access event; 30% control-block leakage integrated over 50 ns |
| Reference | static current-reference seat |
| CABLC | macro `cablc` channel for the complete VDD·I input branch; array row for node capacitance, declared zero in this paper-reproduction preset |
| DSWCT | per-digit current-mirror conduction |
| SINWP-SC | held/live mirror-leg conduction; no fitted hold-cap term |
| PN-ISUB | three internal current branches plus sign-decision per-op energy |
| TMCSA | PH2/PH3 conduction plus one fixed switching event per sensing step and CIM-IO |

The kernel `SarIadc` is energy-silent in this macro; it supplies values and timing only. TMCSA bills the explicit PH2/PH3 branch conduction and 50 fJ per sensing step and CIM-IO for its internal latch, reset nodes, and local switching. An external DOUT register is not included.

## Calibration stages

The cell conductance table and 100 Ω CABLC output resistance are fixed inputs, not calibration variables. The standard solver tool sweeps the iteration counts used by the configured array. The macro's magnitude map is fixed as `[0, 9] -> 0`, `[10, 19] -> 1`, ..., `[70, +inf) -> 7`, so its max-bit rescale factor is the derived value 10 rather than a fitted quantity. The standard ADC probe samples logical weights with shape `[Bw, input_num, output_num]` and inputs with shape `[Bx, 1, input_num]`; each input contains one randomly placed contiguous window of `max_active_num` candidate rows and is never split into planes. Zero-valued samples inside that window remain electrically inactive. After the random phase, the same run checks every reachable signed ideal value within the `[-81, 81]` envelope and conditionally supplements only values below the requested sample threshold; unreachable integers are not treated as coverage failures. Row positions remain independently random and off-window weights still cover the full logical domain. The probe pairs each nominal `I_SUB` observation with the ideal twin's exact result; the validation analysis places each reference at the q99 equal-tail midpoint around the fixed ideal-code boundary.

The adopted read-path parameters were inferred from the CABLC+DSWCT and SINWP-SC+PN-ISUB Fig.18 pairs under the workload declared in `anchors.toml`. That reverse inference is parameter provenance, not a reusable calibration workflow, so no repository tool replays it. At the adopted activity point, the declared array-node capacitance seat is zero because explicit CABLC+DSWCT conduction already slightly exceeds the paired target; this is a paper-reproduction convention, not a claim that the physical capacitances vanish. PN-ISUB uses a 47.665 fJ three-inverter/latch event. TMCSA uses its explicit branch currents, 0.6 ns PH2 and 1.0 ns PH3 windows calibrated at the assumed 0.18:0.30 ratio within the fixed 3.11 ns decision step, and the 50 fJ per-step switching event without a residual multiplier.

Control and reference remain adopted Fig.18 seats. Because the activity probabilities and two remaining read-path parameters are inferred from the same breakdown later used for comparison, agreement is a model-consistency result rather than independent validation.

## ADC margin analysis

The core ADC tool stops after collecting, storing, and summarizing paired observations. This validation owns the fixed Xue2020 interpretation directly in its analysis code; it has no separate analysis config. The script merges any repeated `--input` files, folds signed ideal values to magnitudes, and writes two SVGs: a ridgeline whose vertical grid retains every integer from zero through the theoretical maximum, including empty rows for unreachable values, and a cluster-statistics plot of `min..max`, `p05..p95`, mean, and median. It compares the adjacent exact-value clusters at the fixed `10/20/.../70` boundaries and reports their 95th/99th-percentile separation and midpoint. These boundaries implement `[0, 9] -> 0`, `[10, 19] -> 1`, ..., `[60, 69] -> 6`, and `[70, +inf) -> 7`. Positive and negative samples with the same magnitude occupy the same cluster.

    uv run python -m validations.xue2020jssc.tools.analyze_adc_margin \
        --input <adc-run>/random.pt \
        --input <adc-run>/targeted.pt \
        --log-dir <output-directory> \
        --plot-dir <output-directory>
