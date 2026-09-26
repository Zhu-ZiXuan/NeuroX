# Running calibration tools

Run commands from a source checkout. Install plotting dependencies when using tools that produce figures:

```bash
python -m pip install '.[calib]'
```

## Configuration and invocation

Each calibration command accepts a run configuration with hardware bindings and workload settings. Inspect its options with `--help`.

- `--config` selects the run file.
- `--device` selects the compute device explicitly.
- `--output-dir` selects the parent directory for timestamped runs; `--log-dir` is an alias.
- `--log-level` selects logging verbosity.

Commands that emit a configuration fragment accept `--output` for an additional copy. Commands that plot accept `--plot-dir` for an alternative figures directory.

Paths inside a run file resolve relative to that file. Absolute paths stay absolute. Keep referenced files together when moving an experiment. Stimulus seeds control workload sampling; physical variation also depends on the model's runtime RNG state.

## Results

Each run retains `run.log`, invocation status and parameters in `run.json`, and parsed calibration values in `config.json`. Relative paths in `config.json` retain their meaning against the source config path recorded in `run.json`.

Failed runs preserve completed artifacts and record the exception. A completed process status does not establish scientific acceptance; evaluate the task's reported results and assumptions.

## Calibration order

1. Extract a linearized cell if the selected model needs calibrated tables.
2. Choose the macro's operating points and quantization windows.
3. Characterize ADC input clusters and choose appropriate references.
4. Derive or fit full-resolution macro output scales using those references.

Run files may live anywhere. Paper-specific configurations belong with their [validation campaign](../../validation/campaigns.md) so their provenance can be reproduced.
