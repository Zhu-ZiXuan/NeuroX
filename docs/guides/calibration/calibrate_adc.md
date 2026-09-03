# ADC input characterization

Goal: characterize the nominal ADC input clusters of one CIM-macro operating mode so the converter's reference values or decision boundaries can be chosen from circuit knowledge. The tool reports observations; it does not assume a current or voltage domain, map ideal values into ADC codes, or emit a reference configuration.

```bash
CUDA_VISIBLE_DEVICES=0 python -m neurox.tools.calibrate_adc \
    --config validations/<paper>/tools/calibrate_adc.toml \
    --device cuda \
    --output-dir log/calibration \
    --min-samples-per-ideal-value 65536
```

## Run conditions

One run names one physical macro and one `quantization_mode`. Both paths explicitly pass `adc_active_bits = None`: the physical macro uses its maximum ADC width, while the ideal twin retains every exact integer result without invoking an ADC. A different mode, activation limit, value domain, or circuit operating point requires a different run config.

Use an all-off policy. The probe is intended to expose the deterministic cluster spread produced by the nominal circuit, spatial placement, loading, and IR drop. Fabrication mismatch, runtime noise, stochastic rounding, and other sampled nonidealities would instead make the result conditional on random draws.

The tool derives every integer in the macro's declared weight and input ranges and samples those values uniformly. `active_row_selection = "scattered"` chooses input positions uniformly without replacement; `"contiguous"` chooses a uniformly placed contiguous window. It generates `max_active_num` values independently from their positions and scatters them into the full input. A selected zero is electrically inactive, so the actual active count may be smaller. The input is executed once and is never serialized into a plane axis. `[stimulus.random]` and `[stimulus.target]` carry the independent seeds and batch sizes for the two phases.

Uniform primitive values do not produce uniform exact sums: central ideal values normally occur more often than extremes. Cluster counts are therefore part of the report and must be inspected before an extreme cluster is used to set a boundary.

Every run first performs the unconstrained random scan. It derives the exact reachable set of signed ideal values from the macro's input/weight domains and `max_active_num`; the numeric minimum-to-maximum interval may contain unreachable integers. The tool compares each reachable value's random count with `--min-samples-per-ideal-value`, whose default is 65,536. Values below the threshold are supplemented automatically with discrete conditional sampling. The sampler generates input/weight pairs with the requested dot product; positions remain independently random, and weights outside the selected positions remain uniformly random. Supplementation is rounded up to complete `target.batch_w × output_num` batches, so the final count is never below the requested threshold.

## Paired observations

Every ADC implementation submits its input record through the shared `AdcProber`. The record supplies a unit-bearing name and the scalar decision input: a single-ended converter returns its terminal quantity, while a differential converter returns its positive input minus its negative input. Converter-facing tensors use the macro family's canonical lane-major, scan-minor order, so flattening pairs the input samples elementwise with the ideal twin's exact highest-precision output.

The physical ADC output code is neither recorded nor used. The run is establishing the ADC's operating conditions, so a code produced by provisional references has no calibration meaning. Exact ideal values are likewise left unmodified: no clipping, saturation, magnitude fold, zero-point shift, input-code mapping, target-window shading, or overflow classification occurs. The theoretical exact-result support is derived only from the declared weight domain, input domain, and `max_active_num`; a finite random run is not expected to visit every value in that support.

## Report

`--output-dir` names only the parent directory. The tool creates one `adc_probe_<UTC timestamp>/` child for the run and places every artifact directly inside it:

- `adc_probe.log` — run conditions, coverage planning, final minimum coverage, and random-phase statistics.
- `random.pt` — paired observations from the unconstrained phase.
- `targeted.pt` — only the conditional samples added to fill coverage deficits; it is valid but empty when random sampling already meets the threshold everywhere.

Both data files contain paired one-dimensional CPU tensors: `float32` ADC inputs and exact `int64` ideal values, plus the input quantity name and exact reachable ideal-value set. Load them with `load_adc_probe_data` and concatenate them before validation-specific analysis. Keeping the phases separate preserves the natural random code frequencies while allowing conditional per-code statistics to use the complete dataset.

The core tool deliberately produces no plot. Whether signed ideal values stay separate, fold to magnitudes, map into another logical code, or use a circuit-specific axis is part of the validation interpretation rather than ADC-input collection.

## Downstream analysis

Reference placement is not universal. A full comparator ladder, an internally scaled single reference, and a differential SAR full-scale reference require different circuit reasoning even when their observed clusters look similar. A validation campaign owns any plots and analyzers over the saved observations because it knows the required code transform, reference topology, and decision rule.

The paired data file is the reusable boundary between collection and analysis. It contains observations only, not a chosen reference ladder or an ADC-specific interpretation.
