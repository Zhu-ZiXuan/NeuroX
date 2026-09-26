# Macro rescaling

A macro's `rescale_factors` convert final output codes into integer MAC units at maximum ADC resolution. Each `quantization_mode` selects one reference operating point and its corresponding factor.

Derive the factor analytically when the mapping is exact. Otherwise fit paired physical output codes and exact ideal results through the origin:

$$\mathrm{ideal\ value}pprox\mathrm{macro\ code}	imes\mathrm{rescale\ factor}.$$

## Run a fit

Prepare a run file using the `RescaleFitToolConfig` schema in the [rescale API](../../api/tools.md#neurox.tools.calibration.cim_macro.rescale_fit). It names the macro bindings, stimulus distribution, sample counts, batch sizes, and seed. The path below is a user-created run file:

```bash
uv run python -m neurox.tools.calibration.cim_macro \
    --config rescale_run.toml \
    --device cuda:0
```

`--modes` selects a comma-separated subset. The command writes `rescale.toml`, retaining factors for unselected modes, and per-mode fit plots under `figures/`. Inspect residuals and saturation before adopting the fitted values. ADC reference selection is a separate step and must precede this fit.

## Ideal comparisons

Use a macro's `to_ideal()` method to construct an arithmetic control and program it with the same weights. `adc_active_bits=None` requests maximum physical precision or exact ideal execution; an explicit width applies finite-resolution quantization. The [macro interface](../../api/components.md#neurox.primitive.macro.cim.CimMacro) defines the conversion and programming contract.
