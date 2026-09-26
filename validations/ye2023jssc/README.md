# Ye2023 JSSC validation

This campaign compares the WH-2T1R macro with the measured power breakdown in Fig. 19, using the published configuration without redundant sub-array mapping. Static energy and average power use the paper's 85 ns measurement cycle; the campaign applies its test waveform, while the bundled work preset supplies optimized timing for normal evaluation.

The 50% input-sparsity point is the calibration anchor and the only hard gate. The 87.5% point is an informational extrapolation because the two published breakdowns cannot be reproduced by one activity-independent accounting boundary.

Before reporting, the campaign divides area and leakage by the known macro count, preserving dynamic energy and working duration. It then groups paper components, computes repeat statistics, normalizes by scan count, and converts energy to power for comparison.

Run the record workload with:

```bash
make validate_ye2023jssc DEVICE=cuda:0
```

The command creates `ye2023jssc_<UTC timestamp>/` under `log/validation/ye2023jssc/`, containing `run.log`, `run.json`, `profile.pt`, and `power_breakdown.svg`. The profile maps input sparsity to named observations, retaining `[repeat * n_x, macro_instance]` in collection order. Analysis uses `n_x` to recover repeats. For additional options, invoke `uv run python -m validations.ye2023jssc.validate --device cuda:0`; `--output-dir` selects the parent directory and `--log-level` controls verbosity.

`config.toml` selects `neurox/presets/works/ye2023jssc.toml` through `_neurox_use_preset` and overrides RS-CSA timing with the paper's 20 ns-per-phase waveform. The profiler records each VMM's modeled working duration. Reporting uses `scan_num × 85 ns` as the separate powered window, then normalizes static energy and average power to one macro and one access.

## Reference

```bibtex
@article{ye2023jssc,
    author  = {Wang Ye and Linfang Wang and Zhidao Zhou and others},
    title   = {A 28-nm RRAM Computing-in-Memory Macro Using Weighted Hybrid 2T1R Cell
               Array and Reference Subtracting Sense Amplifier for AI Edge Inference},
    journal = {IEEE Journal of Solid-State Circuits},
    year    = {2023},
    volume  = {58},
    number  = {10},
    pages   = {2839--2848},
    doi     = {10.1109/JSSC.2023.3280357},
}
```
