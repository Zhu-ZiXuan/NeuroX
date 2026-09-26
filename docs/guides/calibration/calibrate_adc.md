# ADC input characterization

Collect nominal ADC input distributions for a configured macro mode, then use circuit-specific analysis to select reference values.

The bundled Xue2020 run file can be invoked from the repository root:

```bash
uv run python -m neurox.tools.calibration.adc \
    --config validations/xue2020jssc/tools/calibrate_adc.toml \
    --device cuda:0 \
    --output-dir log/calibration
```

This is a sampling campaign rather than a quick smoke test. Choose sample and batch counts in the run file, and use `--min-samples-per-ideal-value` to set the required coverage.

## Conditions and coverage

Use an all-off non-ideality policy to study deterministic spread from placement, loading, and IR drop. A new mode, value domain, activation limit, or operating point requires a corresponding run configuration.

The tool samples uniformly from the declared primitive input and weight values. Active-row selection can be scattered or contiguous. Selected zero values remain electrically inactive, so the active count can be below the selection limit.

Uniform operands do not produce uniform dot products. The tool derives the reachable ideal-value set, performs a random phase, and conditionally samples values below the requested minimum coverage. Supplemental sampling uses whole batches and can exceed the requested count. Inspect the reported coverage before using rare clusters to select boundaries.

## Observations

Each ADC input is paired with the ideal twin's exact result. A differential converter contributes its positive-minus-negative input. Physical output codes are not used for reference selection because provisional references would bias the interpretation.

The run writes `random.pt` and `targeted.pt`; the latter can be empty when the random phase meets coverage. Use `load_adc_probe_data` from the [probe-data API](../../api/tools.md#neurox.tools.calibration.adc.data) to load them. Keep the phases separate when analyzing natural sample frequencies; combine them when estimating conditional statistics at each ideal value.

## Reference selection

The tool saves observations rather than a reference configuration. A comparator ladder, a scaled single reference, and a differential full-scale reference require different analyses. Campaign-specific analyzers can fold signed values, compare decision boundaries, and plot margins according to the circuit topology. The data format and collection contracts are documented in the API.
