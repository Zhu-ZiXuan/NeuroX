# Tools

`neurox/tools/` collects offline calibration and analysis scripts that run outside the main forward path. Each tool is a small CLI that consumes a chip TOML and emits a calibration artefact or a diagnostic report.

## Current tools

- `xbar_adc_boundaries.py` — runs the real-vs-ideal xbar comparison over a sweep of inputs and emits a calibrated `[bl_adc]` TOML block (range
  + precision; floor-style boundaries; optional noise injection;
  optional `--visualize` PNG of the signal-vs-code distribution).
- `calculate_xbar_param_1t1r.py` — 1T1R-specific pre-distortion and ADC boundary calculator. Given a chip TOML plus a target programming range, derives the per-state RRAM conductance list whose on-state per-cell current ladder is linear through the series RRAM/NMOS cascade and the matching ADC boundary grid.
- `analyze_xbar_error_1t1r.py` — VMM-accuracy analysis. Builds a noiseless 1T1R xbar from a chip TOML, compares it against its `to_ideal()` reference, and reports per-code error statistics. With `--noise`, also builds five per-category noisy variants (RRAM stuck-at, RRAM programming Gamma, RRAM read noise, NMOS fabrication mismatch, periphery noise) and ranks each category's contribution to ADC-code error.

## Why these are tools, not library code

They are reproducible **offline** flows: a developer runs them once when tuning a chip configuration; the resulting TOML / TXT artefacts then feed back into the main forward path. Keeping them under `tools/` rather than mixing with library code makes the I/O surface explicit (TOML in, TOML / PNG / log out).

## Shared library helpers

`xbar_adc_boundaries.py` also re-exports a few small pure helpers (`floor_boundaries_for_mode`, `compute_max_col_diff_current__uA`, `compute_adc_boundaries__uA`) that other tools and tests use as calibration primitives. They are public because their semantics ("floor-style boundaries at code edges") are part of the ADC family's external contract.

See also:

- `../analog/adc/README.md`
- `../xbar/_1t1r/circuit_core.md`
