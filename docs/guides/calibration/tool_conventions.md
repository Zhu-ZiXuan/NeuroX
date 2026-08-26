# Calibration tool conventions

The rules every calibration command shares, so a run you can drive for one tool you can drive for all.

## The run config holds everything that changes the answer

Anything a re-run must reproduce lives in the run config: the hardware config and policy the tool builds from, the workload distribution, the sweep ranges and thresholds, the random seed, and the computation dtype. Two runs of one config give one answer, whatever the invocation looks like.

The CLI carries only runtime and output controls:

- `--config <run.toml>` — the run config. Required by every tool.
- `--device` — the compute device, on a tool that touches one. It defaults to CPU: no tool picks up a GPU implicitly, so a GPU run says so on the command line.
- `--plot-dir`, `--output`, `--output-dir` — where the optional artifacts go. A tool declares whichever of these it emits.
- `--log-dir` — where the timestamped plain-text run log is written.
- `--log-level` — one of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

A path read from a run config resolves against that config file's own directory; an absolute path stays absolute. So a run config references its neighbours by relative path and moves as a directory.

## Results live in the run log

A tool reports through the logger in a message-only format on stderr, with no level prefix or timestamp decorating a line. A characterization command records measurements and diagnostics; a command that derives a config value prints a directly pasteable TOML fragment under the table named in the fragment's own header comment.

A tool writes files only where its CLI declares them. Calibration commands keep a per-run log under `--log-dir`, so long or staged runs retain their inputs, diagnostics, and emitted fragments without a hand-copied Markdown report.

## Where the run configs live

A calibration run config sits beside the campaign params it seats, under `validations/<paper>/tools/`, and references those params by relative path. The per-paper directory contract, and the provenance tags a seated value carries once it is written back, are in [validation campaigns](../../validation/campaigns.md).

## Calibration order

Each stage consumes what the stage before it seated, so run them in dependency order:

1. **Cell** — the per-cell condensation count and, for an array running a linearized cell, the chord extraction ([solver iteration counts](solver_iteration_counts.md)).
2. **Array solver** — the iteration counts of the DC solve the cell feeds ([solver iteration counts](solver_iteration_counts.md)).
3. **Quantization mode set** — the macro's numerical operating windows ([macro calibration](calibrate_macro.md)).
4. **ADC input characterization** — the nominal input clusters used to choose the mode's reference values manually ([ADC input characterization](calibrate_adc.md)).
5. **Macro rescale factor** — the per-mode recovery coefficient derived from an exact declared code mapping, or fitted from complete macro outputs when that relationship is empirical ([macro calibration](calibrate_macro.md)).

The direction is fixed: a component-level numerical solve is calibrated before the composite solve that depends on it, and the ADC references are chosen from the characterized physical input before the macro's output-code rescale is fitted.
