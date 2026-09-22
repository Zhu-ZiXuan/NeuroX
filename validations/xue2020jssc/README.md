# xue2020jssc validation

This campaign validates one 256×512 1T1R sub-array against the paper's simulated 32.0625 pJ/access target and Fig.18 energy breakdown.

## Run

```bash
make validate_xue2020jssc DEVICE=cuda:0
```

The command creates `xue2020jssc_<UTC timestamp>/` under `log/validation/xue2020jssc/`, containing `run.log`, `run.json`, raw named observations in `profile.pt`, and `energy_breakdown.svg`. `--output-dir` selects another parent directory; `--log-level` controls message-only logging.

`config.toml` selects the bundled `neurox/presets/works/xue2020jssc.toml` design point through `_neurox_use_preset`.

Before reporting, the campaign divides area and leakage by the known macro count, preserving dynamic energy and working duration. It then groups paper components, averages within experimental repeats, and converts complete-VMM energy to the per-scan basis.

## Files

- `config.toml`: campaign binding to the bundled work preset.
- `neurox/presets/works/xue2020jssc.toml`: bundled model parameters and their provenance.
- `policy.toml`: disabled nonideality policy.
- `anchors.toml`: paper targets, accounting conventions, and workload distribution.
- `validate.py`: energy campaign and breakdown plot.
- `tools/calibrate_adc.toml`: ADC-input sampling campaign.
- `tools/analyze_adc_margin.py`: fixed-boundary reference analysis and plots.

## Model

Two magnitude digits and two polarities map 128 logical columns onto 512 physical columns. A 32:1 column MUX gives four parallel readout lanes. DSWCT applies weight-digit ratios, SINWP-SC accumulates input-bit contributions, PN-ISUB produces sign and magnitude, and TMCSA quantizes the magnitude.

For ADC resolution `b`:

```text
detect(b) = t_settle__ns + b * latency_per_bit__ns
access_latency(b) = (x_bit_num - 1) * t_sample__ns + detect(b)
```

The profiler records each VMM's modeled working duration. Reporting uses the paper's operating period from `anchors.toml`, with `scan_num × 50 ns` as a separate powered window. Export retains `[repeat * n_x, macro_instance]`; analysis uses `n_x` to recover repeats and normalizes static energy to one macro and one access before pooling rounds.

| Fig.18 slice | Model contribution |
| --- | --- |
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

```bash
uv run python -m validations.xue2020jssc.tools.analyze_adc_margin \
    --input <adc-run>/random.pt \
    --input <adc-run>/targeted.pt \
    --log-dir <output-directory> \
    --plot-dir <output-directory>
```

## Reference

```bibtex
@article{xue2020jssc,
    author  = {Xue, Cheng-Xin and Chen, Wei-Hao and Liu, Je-Syu and Li, Jia-Fang and Lin, Wei-Yu and Lin, Wei-En and
                Wang, Jing-Hong and Wei, Wei-Chen and Huang, Tsung-Yuan and Chang, Ting-Wei and
                Chang, Tung-Cheng and Kao, Hui-Yao and Chiu, Yen-Cheng and Lee, Chun-Ying and King, Ya-Chin and
                Lin, Chrong-Jung and Liu, Ren-Shuo and Hsieh, Chih-Cheng and Tang, Kea-Tiong and Chang, Meng-Fan},
    title   = {Embedded 1-Mb ReRAM-Based Computing-in-Memory Macro With Multibit Input
                and Weight for CNN-Based AI Edge Processors},
    journal = {IEEE Journal of Solid-State Circuits},
    year    = {2020},
    month   = {Jan},
    volume  = {55},
    number  = {1},
    pages   = {203--215},
    doi     = {10.1109/JSSC.2019.2951363},
}
```
