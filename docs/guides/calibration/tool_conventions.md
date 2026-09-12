# Calibration tool conventions

The rules every calibration command shares, so a run you can drive for one tool you can drive for all.

## Independent task commands

Each concrete calibration package owns its CLI entry point. Shared configuration loading, logging, and artifact handling are provided by the [offline tool API](../../api/tools.md); the parent calibration package supplies reusable capabilities.

## Configuration and invocation

The run config holds the hardware bindings and workload parameters. Task-specific CLI selectors, such as a mode subset or minimum coverage, are recorded alongside it for each invocation. The shared arguments are:

- `--config <run.toml>` — required run configuration.
- `--device` — required compute device; every task chooses its device explicitly.
- `--output-dir` — parent directory for independent timestamped runs; `--log-dir` is an alias.
- `--log-level` — one of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

A task that emits a configuration fragment accepts `--output` for an additional copy. A task that emits figures accepts `--plot-dir` to override its run-local figures directory.

A path read from a run config resolves against that config file's own directory; an absolute path stays absolute. Keep the referenced hardware, policy, and distribution files with the run config when moving an experiment. Workload seeds control stimulus sampling; physical nonideality draws follow the model's separate runtime randomness.

## Results and artifacts

A command reports through message-only logging on stderr and in its run's `run.log`. The run also contains `run.json` with invocation parameters, execution status, and elapsed time, and `config.json` with parsed calibration values. The latter is a record of the original configuration: relative paths are interpreted against the source config path recorded in `run.json`.

Configuration fragments and figures are retained inside the run directory by default. A fragment is also printed to the log for inspection and manual seating. Failed runs retain completed artifacts and record the exception; a completed execution status describes process completion, while scientific acceptance is part of the task's result.

## Where the run configs live

A calibration run config sits under `validations/<paper>/tools/` and references its campaign's `config.toml` by relative path. That binding selects the bundled `neurox/presets/works/<paper>.toml` design point through `_neurox_use_preset`. The campaign file contract, and the provenance tags a seated value carries once it is written back, are in [validation campaigns](../../validation/campaigns.md).

## Calibration order

Each stage consumes what the stage before it seated, so run them in dependency order:

1. **Cell linearization** — for an array running a linearized cell, extract the chord from converged detailed-cell DCOPs ([cell linearization and solver contract](solver_tolerances.md)).
2. **Quantization mode set** — the macro's numerical operating windows ([macro calibration](calibrate_macro.md)).
3. **ADC input characterization** — the nominal input clusters used to choose the mode's reference values manually ([ADC input characterization](calibrate_adc.md)).
4. **Macro full-resolution rescale factor** — the per-mode recovery coefficient at `adc_bits`, derived from an exact declared code mapping or fitted from complete macro outputs when that relationship is empirical ([macro calibration](calibrate_macro.md)).

The ADC references are chosen from the characterized physical input before the macro's output-code rescale is fitted.
