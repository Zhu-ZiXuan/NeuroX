# Generic ADC calibration tools

Goal: calibrate the ADC operating points of any registered CIM macro without scheme-specific tool code. The package `neurox/tools/calibrate_adc/` holds three config-driven CLIs built on two generic seams: the `CimMacro` registry (the tool TOML names a macro config/policy file pair; `CimMacroConfig.from_file` + `CimMacro.from_config` resolve the concrete tile) and the `AdcProber` calibration channels (`adc.convert` on the physical tile, `adc.ideal_vmm` on its `to_ideal()` twin). The dual probed run serializes every WL plane over the macro's sub-phase axis (at most `max_active_rows` live rows per conversion, rows outside the window zeroed) before both the physical and the ideal drive, so calibration samples are captured under the same per-conversion masked drive the runtime engine layer applies. The tools reference no scheme-specific symbols, and a scheme provides only registered classes and run-config TOMLs beside its params.

All three follow the [tool conventions](README.md#tool-conventions) (`--config`, `--device`, `--plot-dir`, `--log-level`), and additionally take `--log-dir` (per-run log file, default `log/calibration/`) with figures defaulting to `log/calibration/figures/`.

## `calibrate_adc.rescale_fit` — per-(mode, bits) rescale

Fits the scalar `rescale_factor` of the consumer model $|M_{\mathrm{ideal}}| \approx \mathrm{code} \cdot \mathrm{rescale\_factor}$ per ADC operating mode:

1. Build the physical tile from the tool TOML's `[macro]` file references (all-off policy) and its lossless `to_ideal()` twin.
2. For each `[stimulus]` combo, program the same random ternary weights into both tiles and drive the same random binary WL batches; the physical VMM runs at the mode under fit, the ideal at the lossless `adc_bits = 0` sentinel.
3. Pair the `adc.convert` / `adc.ideal_vmm` streams positionally and element-wise (the macro layout contract preserves logical-column order through its readout reshapes).
4. Per mode, keep only the samples inside the mode's fit window: drop pairs with $|M_{\mathrm{ideal}}| > \mathrm{range}$ on the ideal axis and drop top-code-saturated pairs (both dropped counts are logged), then solve the zero-through-origin least squares in float64.

The mode set comes from the [mode-set TOML](#calibrate_adcmode_derive-mode-set-from-per-layer-ranges) the run config points at — each `[[modes]]` record supplies the mode's `range` for the ideal-axis filter. Output: an `[[adc_calibration]]` TOML fragment (one record per (mode, bits), pasted nested under the macro section) plus a per-mode fit plot (code vs ideal + fitted line).

`--modes m[,m...]` narrows a run to a subset of the mode set: each mode re-runs the full stimulus battery, so per-mode runs bound single-command runtime; the emitted fragments concatenate.

## `calibrate_adc.threshold_probe` — analog grid sweep + threshold placement

Probes the analog band the ADC input sees at every integer per-conversion MAC magnitude and places the mid-point threshold ladder:

1. Build the same physical/ideal pair.
2. Run the controlled-stimulus battery: a deterministic single-cell-LSB count grid realizing every $|M| \in [0, m_{\max}]$ under full WL drive (walked over column-offset patterns on narrow tiles; `grid_col_stride` dilutes the programmed columns on wide tiles), count-capped random single-sign block patterns crossed with random drive densities (`full_drive_caps` selects which caps also run a full-drive exact-count element), and optional dense saturating columns. The battery must stay inside the workload envelope the macro's DC solve converges on — the dilution/full-drive knobs exist because a fully-dense full-drive extreme outside a tile's convergent envelope would contaminate the observed bands with a KCL-violating iteration fixed point (verify the envelope with the solver-iteration sweep before enabling the extremes).
3. Pair each conversion's captured analog input (`adc.convert`) with its realized ideal $|M|$ (`adc.ideal_vmm`); pool into per-$|M|$ bands $[\mathrm{lo}(k), \mathrm{hi}(k)]$ (magnitudes above the top code fold into the top band).
4. Per mode in the mode set (each mode's grid top $m_{\max} = \lceil \mathrm{range} \rceil$ comes from its `[[modes]]` record), place $t_k = \tfrac{1}{2}(\mathrm{hi}(k) + \mathrm{lo}(k+1))$ and report the band margins $\mathrm{lo}(k+1) - \mathrm{hi}(k)$ — the minimum margin is the headline; a negative margin means adjacent bands overlap and the placement is invalid at that boundary. A pooled linear $I(M)$ fit and a strict band-mean monotonicity check accompany the report.

Output: a `ref_levels__uA` / `i_refs__uA` row fragment (row index = `adc_mode`) plus figures (grid curve with bands and thresholds per mode; per-mode margin bars).

Capture staging bounds single-command runtime on large batteries: the element list is deterministic for a given config, so `--element-range a:b` probes a contiguous slice, `--capture-out part.pt` saves that slice's pooled streams and defers placement, and a final run merges every `--capture-in` part ahead of its own slice before placing the ladder (the log records the merged provenance).

## `calibrate_adc.mode_derive` — mode set from per-layer ranges

Derives the ADC operating-mode set for a deployment from a neutral per-layer range TOML: one entry per layer, `"layer.name" = { range = <float>, signed = <bool> }`. Extracting the ranges from a training checkpoint is a consumer-side step; the tool reads only this mapping.

1. Split the layers into signed / unsigned groups by their `signed` flag.
2. Cluster the range values within each group (deterministic 1-D relative-gap agglomeration, capped per group); enumerate the clusters as global modes.
3. Emit the mode-set TOML: `[[modes]]` records (`adc_mode`, `signed`, `range`, `layer_num`) plus a `[layers]` `layer -> adc_mode` map.

The mode-set TOML is the single source consumed downstream — `threshold_probe` reads each mode's grid top from it and `rescale_fit` reads its mode list and per-mode fit windows from it. A cluster plot accompanies the output. CPU-only; the tool opts out of `--device`.

## Run configs

A scheme keeps its run configs beside its params (e.g. `calibrate_rescale.toml` / `calibrate_threshold.toml` / `calibrate_modes.toml` in the scheme's `params/` directory). The `[macro].config_files` list merges first-wins, so a geometry overlay can precede the scheme default.

---

- See also: [rescale convention](../../reference/primitive/macro/cim/README.md#output-rescale), [prober internals](../../internals/common/prober.md)
