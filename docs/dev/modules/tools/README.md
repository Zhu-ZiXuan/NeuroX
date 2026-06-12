# Tools

`neurox/tools/` collects offline calibration and analysis scripts that run outside the main forward path. Each tool is a small CLI that consumes a chip TOML and emits a calibration artefact or a diagnostic report.

## Current tools

All tools are **config-driven**: each takes a single `--config <run.toml>`
plus a handful of runtime / output flags (`--device`, `--plot-dir` /
`--plot` / `--output`, `--log-level`). Workload, sweep, RNG seed, dtype
and other inputs live in the TOML config — typically wrapping a chip
preset via ``[xbar]._neurox_use``. Sample TOMLs sit beside each tool
under `example/config/` and bear the tool's module path
(e.g. `example/config/xbar_adc_statistic.toml`).

The ADC-side tools form a **two-stage pipeline** with no overlap:

| Tool | What it answers |
|---|---|
| `xbar_adc.statistic` | "What's the ADC analog-input distribution?" — probes `v_diff` and recommends `[-A, A]` input-range candidates at a ladder of clip rates. Input to `statistic` → chosen via `[plot]`/`[statistic]` knobs; output is the `v_refs` list to paste into the chip's ADC config. |
| `xbar_adc.calibrate` | "What's the recovery rescale at a fixed (mode, bits)?" — runs an all-off xbar vs ideal twin and fits the **scalar** `rescale_factor` (zero-through-origin LS) per `adc_mode`. Output is the `[[xbar.adc_calibration]]` block. |

The other tools:

- `calculate_1t1r_states` — single-cell 1T1R state-map optimizer. Builds nominal RRAM/NMOS device models and solves a one-cell KCL so RRAM states produce a linear cell-current ladder. Emits `rram_g_max__uS` and `state_to_g_map__uS`.
- `xbar_tia.optimize` — TIA-design exploration / scoring under a chip + workload TOML (curve grid + slice plots).
- `solver_calibrate.tia` / `.nested` / `.full_jacobian` — per-solver-family iteration-count calibration via step-ratio plateau detection.

## Why these are tools, not library code

They are reproducible **offline** flows: a developer runs them once when tuning a chip configuration; the resulting TOML / TXT artefacts then feed back into the main forward path. Keeping them under `tools/` rather than mixing with library code makes the I/O surface explicit (TOML in, TOML / PNG / log out).

## Shared library helpers

`_config.py` provides the shared CLI helpers (`add_standard_args`,
`load_tool_config`, `resolve_relative_path`, `setup_logging`) that every
tool's `main()` calls — see [`_config.md`](_config.md) and
[`logging.md`](logging.md).

See also:

- `docs/dev/modules/analog/adc/README.md`
- `docs/dev/modules/xbar/_1t1r/circuit_core.md`
