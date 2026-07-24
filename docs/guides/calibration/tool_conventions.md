# Calibration tool conventions

Calibration tools run outside the forward path. Each command consumes a run
config and emits a calibration artifact or diagnostic report.

## Configuration

Inputs that affect the result belong in the run config: hardware configuration,
policy, workload distribution, sweep ranges, random seed, and computation dtype.
The CLI carries runtime and output controls:

- `--config <run.toml>` selects the run config.
- `--device` selects the compute device when the tool supports more than one.
- `--plot`, `--plot-dir`, `--output`, or `--output-dir` selects an optional
  artifact destination.
- `--log-level` selects one of `DEBUG`, `INFO`, `WARNING`, `ERROR`, or
  `CRITICAL`.

A path read from a run config is resolved relative to that config file. Absolute
paths remain absolute.

## Output

Tools report human-readable results through the `neurox.tools` logger. Logging
uses a message-only format on stderr so emitted config fragments remain directly
usable. A tool writes a file only when its CLI declares an output option.

## Ordering

Calibrate a component-level numerical solve before a composite solve that
depends on it. Select an ADC analog range before fitting a recovery rescale for
that range.
