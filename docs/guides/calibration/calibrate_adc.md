# ADC calibration

Goal: seat the ADC operating points of a CIM macro — the quantization mode set, the analog threshold ladder, and the per-mode rescale factor — so a run's output codes carry the MAC values the model expects.

The commands live in `neurox.tools.calibrate_adc` and are scheme-independent. They resolve the macro through the macro registry, so a scheme contributes run-config TOMLs and nothing else; importing `neurox` registers every bundled macro before dispatch.

The two probing steps build the configured macro alongside its lossless `to_ideal()` twin. The physical macro's ADC input is captured through a probe while the twin's return value is the ideal view directly, and both macros are driven over the macro's sub-phase axis — at most `max_active_num` live rows per conversion, the rest zeroed — so a calibration sample is taken under the same masked drive the engine applies at run time. Run them under the all-off policy: a nonideality left on turns a placement into a sample of one random draw.

All three follow the [tool conventions](tool_conventions.md) and add `--log-dir` for the per-run log file. Each command's own mechanism — the stimulus battery, the pairing rule, the fit target — is in its module docstring, reachable with `--help`.

## Step 1 — derive the mode set

`mode_derive` turns a per-layer range mapping into the global quantization mode set. Its input is neutral: one entry per layer, `"layer.name" = { range = [lo, hi] }`, giving the layer's inclusive design range in MAC units. Extracting those ranges from a training checkpoint is a consumer-side step; the tool reads only the mapping.

```bash
python -m neurox.tools.calibrate_adc.mode_derive \
    --config validations/<paper>/tools/<mode_derive_run>.toml \
    --output validations/<paper>/tools/modes.toml
```

Each layer maps onto the canonical integer window covering its range — unsigned for a non-negative range, the smallest mid-zero window otherwise — and the windows are clustered within their shape group, one mode per cluster sized to its largest member. The emitted TOML carries a `[[modes]]` record per mode (`quantization_mode`, `quantization_input_range`, `layer_num`) plus a `[layers]` map from layer to mode, and a cluster plot lands under `--plot-dir`.

That TOML is the single source the next two steps read, so write it where their run configs point (`modes_file`). The tool is CPU-only and takes no `--device`.

A scheme with one fixed operating mode writes its mode-set TOML by hand instead and skips this step.

## Step 2 — place the threshold ladder

`threshold_probe` sweeps a controlled stimulus battery, pools the analog input observed at every integer ADC input code into a band, and places the mid-point ladder between adjacent bands.

```bash
python -m neurox.tools.calibrate_adc.threshold_probe \
    --config validations/<paper>/tools/calibrate_threshold.toml --device cuda:0
```

Read the margins first. The headline is the minimum band margin; a negative margin means two adjacent bands overlap and the placement is invalid at that boundary, which no later stage can repair. A pooled linear fit and a band-mean monotonicity check accompany the report as sanity signals.

Keep the battery inside the workload envelope the macro's DC solve converges on. A fully dense, fully driven extreme outside that envelope contaminates the observed bands with a KCL-violating iteration fixed point rather than a physical current, so `grid_col_stride` dilutes the programmed columns on a wide array, `full_drive_caps` selects which count caps also run a full-drive element, and `include_saturating` gates the dense saturating columns. Confirm the envelope with the [solver iteration sweep](solver_iteration_counts.md) before enabling the extremes.

The output is a single `i_refs__uA` bank, one row per mode with the row index as `quantization_mode`, pasted into the macro's `reference_config`. That reference block is the whole ladder source the ADC reads per call, at the macro's maximum resolution; every lower bit width runs against the same full ladder. Figures per mode — the grid curve with its bands and thresholds, and the margin bars — land under `--plot-dir`.

## Step 3 — fit the rescale factor

`rescale_fit` fits the per-mode `max_bits_rescale_factor`, the coefficient carrying a macro output code back into ideal-macro codes. Run it after the ladder is in the config, since it measures the codes that ladder produces.

```bash
python -m neurox.tools.calibrate_adc.rescale_fit \
    --config validations/<paper>/tools/calibrate_rescale.toml --device cuda:0
```

The fit runs at the macro's `adc_max_bits` only; every lower width follows the family bit-width law from that one factor and needs no fit of its own. Watch the two logged drop counts per mode — pairs outside the mode's input code range and top-code-saturated pairs — since a mode that drops most of its battery was fitted on a thin sample.

The output is a `[[modes]]` fragment, one table per mode carrying the canonical window, the ADC input code range, and the fitted factor, to be pasted nested under the macro's own section. `--modes m[,m...]` narrows a run to a subset of the mode set; each mode re-runs the full stimulus battery, so per-mode runs bound single-command runtime and the emitted fragments concatenate.

## Staging a long capture

A large battery can outlast a single command. The battery's element list is deterministic for a given config, so `threshold_probe` splits: `--element-range a:b` probes a contiguous slice, `--capture-out part.pt` saves that slice's pooled streams and skips placement, and a final run merges every `--capture-in` part ahead of its own slice before placing the ladder over the union. The log records the merged provenance.

## Naming the macro

A probing run config names its macro in `[macro]`: the config files, the section inside them, the policy file, and its section, all by path relative to the run config ([tool conventions](tool_conventions.md)). The `config_files` list merges first-wins, so a geometry overlay can precede the scheme default without editing it.
