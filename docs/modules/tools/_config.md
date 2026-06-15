# `neurox/tools/_config.py`

Shared CLI helpers for the config-driven tools under `neurox.tools`. Every tool's `main()` consumes:

- A `--config <run.toml>` whose dataclass schema lives with the tool.
- Optional output flags (`--plot-dir` / `--plot` / `--output`).
- `--device` (default `cpu`; opt out with `device=False` for fixed-precision CPU-only tools).
- `--log-level` (default `INFO`; one of `DEBUG / INFO / WARNING / ERROR / CRITICAL`).

## Surface

- `add_standard_args(parser, *, device=True, plot_dir=False, plot_file=False, output_file=False)` — append the standard flags. Pass `device=False` for tools that have no GPU code path (e.g. `calculate_1t1r_states` runs entirely in float64 on CPU) so the CLI does not expose a knob the tool would silently ignore.
- `setup_logging(level_name)` — configure `neurox.tools` logging from `args.log_level`. Wraps `neurox.tools.logging.config_tool_logging`.
- `load_tool_config(cls, config_path)` — thin wrapper over `dataclass_from_file(cls, config_path)`. Exists so tool-side imports stay shallow.
- `resolve_relative_path(path, base)` — resolve a TOML-supplied path against `base.parent`; absolute paths and `None` are returned unchanged.

## Conventions

- Tools take **every** input that affects the result via TOML — chip preset (`[xbar]._neurox_use`), workload, sweep ranges, RNG seed, dtype. The CLI carries only runtime / output knobs.
- Default device is CPU. There is no implicit GPU pickup; the user must pass `--device cuda:N` explicitly.
- `--log-level` is enforced via argparse `choices` so a typo errors at parse time rather than at first `getattr(logging, ...)` lookup.
