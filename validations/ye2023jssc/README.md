# Ye2023 JSSC validation

This campaign compares the WH-2T1R macro with the measured power breakdown in Fig. 19. It models the published configuration without redundant sub-array mapping. Static energy and average power use the validation-only 85 ns measurement cycle, and the campaign config applies the paper's test waveform so one row-scan access has the same duration. The bundled work preset retains the optimized timing used for normal evaluation.

The 50% input-sparsity point is the calibration anchor and the only hard gate. The 87.5% point is an informational extrapolation because the two published breakdowns cannot be reproduced by one activity-independent accounting boundary.

Run the record workload with:

```bash
make validate_ye2023jssc DEVICE=cuda:0
```

The command creates one `ye2023jssc_<UTC timestamp>/` directory under `log/validation/ye2023jssc/`, containing `run.log`, `run.json`, and `power_breakdown.svg`. `--output-dir` selects another parent directory; `--log-level` controls the shared message-only logger.

`config.toml` selects the bundled `neurox/presets/works/ye2023jssc.toml` design point through `_neurox_use_preset`, then overrides its optimized RS-CSA timing with the 20 ns-per-phase test waveform. The validation checks that the resulting four-bit access duration equals the 85 ns measurement cycle.
