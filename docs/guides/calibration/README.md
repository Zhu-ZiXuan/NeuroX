# Calibration & tools

The offline tools run outside the main forward path. Each is a small CLI that consumes a chip TOML and emits a calibration artefact (TOML / TXT) or a diagnostic report (PNG / log). A developer runs them once when tuning a chip configuration; the resulting artefacts feed back into the main forward path. The chip-specific calibrators (ADC range / rescale, state map) live with their scheme under that scheme's `tools/` package; `neurox/tools/` holds the scheme-agnostic tools — the iteration-count calibrators (`neurox.tools.calibrate_cell`, `neurox.tools.calibrate_solver`, driven by generic config fragments any scheme supplies via `_neurox_use`) and the [generic ADC calibration package](calibrate_adc.md) (`neurox.tools.calibrate_adc`), which resolves any registered CIM macro through the registry and probes it through the `AdcProber` channels.

All tools here are *calibration* flows that produce the parameters tying an abstract NeuroX model to a specific chip: the ADC input range, the per-operating-point ADC `rescale_factor` table, the single-cell 1T1R state map, the array-solver iteration counts, and the per-cell access-node condensation count. Physical / circuit design parameters come from the circuit design flow and enter simulations via config; NeuroX tools calibrate only the simulator's own numerical constants.

**Calibration order.** Component-level solves are calibrated first — the cell condensation (`calibrate_cell`) — then the array-solver constants (`calibrate_solver`) on a config-complete xbar.

## Tool conventions

All tools are **config-driven**: each takes a single `--config <run.toml>` plus a handful of runtime / output flags. Every input that affects the result lives in the TOML — chip preset (typically wrapping a preset via `[cim_macro]._neurox_use`), workload, sweep ranges, RNG seed, dtype. The CLI carries only runtime / output knobs:

- `--config <run.toml>` — the run config; its dataclass schema lives with the tool.
- `--device` — compute device, default `cpu`. There is no implicit GPU pickup; pass `--device cuda:N` explicitly. Tools with no GPU code path (e.g. the 1T1R state map, which runs entirely in float64 on CPU) opt out of this flag.
- `--plot-dir` / `--plot` / `--output` / `--output-dir` — output destinations, as applicable per tool.
- `--log-level` — one of `DEBUG / INFO / WARNING / ERROR / CRITICAL`, default `INFO`. Enforced via argparse `choices` so a typo errors at parse time.

**Sample-TOML naming.** A ready-to-edit sample config sits beside each tool's scheme config dir and bears the tool's module path, e.g. `<scheme>/config/calibrate_adc_statistic.toml`.

**Shared helpers.** `neurox/tools/_config.py` provides the CLI helpers every tool's `main()` calls:

- `add_standard_args(parser, *, device=True, plot_dir=False, output_file=False, output_dir=False)` — append the standard flags. Pass `device=False` for tools with no GPU code path so the CLI does not expose a knob the tool would silently ignore.
- `load_tool_config(cls, config_path)` — thin wrapper over `cls.from_file(config_path)`, so tool-side imports stay shallow.
- `resolve_relative_path(path, base)` — resolve a TOML-supplied path against `base.parent`; absolute paths and `None` are returned unchanged.
- `setup_logging(level_name)` — configure `neurox.tools` logging from `args.log_level`.

**Logging format.** `neurox/tools/_logging.py` centralises the logging setup so every tool emits in the same paste-ready format: a plain `"%(message)s"` (no level / timestamp / logger-name prefix) on stderr, so a tool's output lines drop straight into a TOML / config file. The setup is idempotent. Tools never call `logging.basicConfig(...)` directly.

## Two-stage ADC pipeline

The ADC-side tools form a two-stage pipeline with no overlap — first pick the analog input range, then fit the recovery rescale at a fixed operating point:

| Stage | Tool | What it answers |
|---|---|---|
| 1 | `calibrate_adc.statistic` | "What is the ADC analog-input distribution?" — probes $v_\mathrm{diff}$ and recommends $[-A, A]$ input-range candidates at a ladder of clip rates. Output is the `v_refs` list to paste into the chip's ADC config. |
| 2 | `calibrate_adc.calibrate` | "What is the recovery rescale at a fixed (mode, bits)?" — runs an all-off cim_macro against an ideal twin and fits the scalar `rescale_factor` (zero-through-origin least squares) per `adc_mode`. Output is the `[[cim_macro.adc_calibration]]` block. |

## Per-tool guides

- [Generic ADC calibration](calibrate_adc.md) — scheme-agnostic `neurox.tools.calibrate_adc` package: rescale fit, analog threshold probe with band margins, and mode-set derivation from per-layer ranges.
- [ADC input-range probing](adc_range_probing.md) — stage 1: probe $v_\mathrm{diff}$ and recommend $[-A, A]$ range candidates.
- [ADC rescale calibration](adc_rescale.md) — stage 2: fit the per-operating-point `rescale_factor` table.
- [1T1R state-map optimization](state_map.md) — single-cell state map: solve one-cell KCL so RRAM states give a linear cell-current ladder; emits `rram_g_max__uS` and `state_to_g_map__uS`.
- [Solver iteration-count calibration](solver_iteration_counts.md) — fixed iteration counts for the array solver and the per-cell access-node condensation, via step-ratio plateau detection. The calibrators are `neurox.tools.calibrate_cell` for the per-cell condensation and `neurox.tools.calibrate_solver` for the array solver; one calibrate-cell run also emits a linearized-cell config fragment.

---

- See also: [rescale convention](../../reference/primitive/macro/cim/README.md#output-rescale), [solver iteration internals](../../internals/primitive/xbar/solver.md), [config & policy](../../internals/config_and_policy.md)
