# Tools

`neurox/tools/` collects offline calibration and analysis scripts that run outside the main forward path. Each tool is a small CLI that consumes a chip TOML and emits a calibration artefact or a diagnostic report.

## Current tools

- `calculate_1t1r_states.py` — single-cell 1T1R state-map optimizer. Builds nominal RRAM/NMOS device models and solves a one-cell KCL so RRAM states produce a linear cell-current ladder. Emits `rram_g_max__uS` and `state_to_g_map__uS`; ignores ADC and xbar-level rescale calibration.
- `xbar_adc_boundaries.py` — runs the real-vs-ideal xbar comparison over a sweep of inputs and emits a calibrated `[bl_adc]` TOML block (range
  + precision; floor-style boundaries; optional noise injection;
  optional `--visualize` PNG of the signal-vs-code distribution).
- [`xbar_adc/`](xbar_adc/README.md) — paired xbar-level ADC range / calibration CLIs. `xbar_adc.statistic` probes the ADC analog-input distribution and recommends `[-A, A]` candidates; `xbar_adc.calibrate` fits the scalar `rescale_factor` for a configured `adc_mode`. Both share `_sampling.py` (distribution loader, samplers, all-off xbar builder) and operate strictly at the xbar layer.

## Why these are tools, not library code

They are reproducible **offline** flows: a developer runs them once when tuning a chip configuration; the resulting TOML / TXT artefacts then feed back into the main forward path. Keeping them under `tools/` rather than mixing with library code makes the I/O surface explicit (TOML in, TOML / PNG / log out).

## Shared library helpers

`xbar_adc_boundaries.py` also re-exports a few small pure helpers (`floor_boundaries_for_mode`, `compute_max_col_diff_current__uA`, `compute_adc_boundaries__uA`) that other tools and tests use as calibration primitives. They are public because their semantics ("floor-style boundaries at code edges") are part of the ADC family's external contract.

`logging.py` exposes `config_tool_logging(level=logging.INFO)` — the single entry point every tool calls in `main()` to install the standard CLI logging format. See [`logging.md`](logging.md).

See also:

- `docs/dev/modules/analog/adc/README.md`
- `docs/dev/modules/xbar/_1t1r/circuit_core.md`
