# xue2020jssc validation

This campaign validates one 256×512 1T1R sub-array against the paper's simulated 32.0625 pJ/access target and Fig.18 energy breakdown.

## Run

    make validate_xue2020jssc

Results are written to `log/validation/xue2020jssc/validation.log` and `energy_breakdown.svg` by default.

## Files

- `params.toml`: model parameters and their provenance.
- `policy.toml`: disabled nonideality policy.
- `anchors.toml`: paper targets, accounting conventions, and workload distribution.
- `validate.py`: energy campaign and breakdown plot.
- `tools/calibrate_solver.toml`: solver iteration sweep.
- `tools/calibrate_adc.toml`: ADC-input sampling campaign.
- `tools/analyze_adc_margin.py`: fixed-boundary reference analysis and plots.

## Model

Two magnitude digits and two polarities map 128 logical columns onto 512 physical columns. A 32:1 column MUX gives four parallel readout lanes. DSWCT applies weight-digit ratios, SINWP-SC accumulates input-bit contributions, PN-ISUB produces sign and magnitude, and TMCSA quantizes the magnitude.

For ADC resolution `b`:

    detect(b) = t_settle__ns + b * latency_per_bit__ns
    access_latency(b) = (x_bit_num - 1) * t_sample__ns + detect(b)

The preset uses a 14.60 ns maximum-resolution access latency and the paper's 50 ns operating period as the leakage integration window.

| Fig.18 slice | Model contribution |
|---|---|
| Control | per-access event and leakage |
| Reference | TMCSA reference leakage |
| CABLC | complete input-branch conduction and array-node capacitance |
| DSWCT | weight-digit mirror conduction |
| SINWP-SC | accumulated input-bit conduction |
| PN-ISUB | three branches and sign-decision event |
| TMCSA | PH2/PH3 conduction and per-bit switching |

Control and Reference adopt the reported Fig.18 shares. CABLC+DSWCT and SINWP-SC+PN-ISUB are compared as pairs because the paper does not publish the internal node voltages needed to separate each series branch. The activity probabilities and remaining read-path parameters were inferred from those same pair targets, so the result is a model-consistency check rather than an independent validation.

## ADC references

The ADC campaign samples the configured nine-row window and supplements underrepresented reachable ideal values. The analyzer folds signed values to magnitudes, plots their input distributions, and estimates references at the fixed `10/20/.../70` ideal-value boundaries.

    uv run python -m validations.xue2020jssc.tools.analyze_adc_margin \
        --input <adc-run>/random.pt \
        --input <adc-run>/targeted.pt \
        --log-dir <output-directory> \
        --plot-dir <output-directory>
