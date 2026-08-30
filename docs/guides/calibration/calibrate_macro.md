# Macro calibration

Macro calibration determines the relationship between a physical macro's final output codes and exact integer MAC values. ADC reference or boundary placement is a separate calibration task.

## Determine the rescale factors

Every reference operating point has one corresponding entry in `rescale_factors`. The entry describes how many MAC units one final macro output code represents at the macro's maximum ADC resolution, `adc_bits`. The tuple index is `quantization_mode`.

When the macro implements an exact integer mapping, derive the factor from that mapping and store it directly. Use `rescale_fit` only when the complete macro transfer makes the relationship empirical. The tool compares the physical macro's final output at `adc_bits` with its ideal twin at `adc_active_bits = 0`; it does not inspect the ADC type or use an ADC probe.

```bash
python -m neurox.tools.calibrate_macro.rescale_fit \
    --config validations/<paper>/tools/calibrate_macro_rescale.toml \
    --device cuda:0
```

Logical weights have shape `[Bw, input_num, output_num]`. Logical inputs have shape `[Bx, 1, input_num]` and are split into legal `max_active_num` planes before both macros run. For every selected mode, the zero-through-origin fit estimates

$$\mathrm{ideal\ value}\approx\mathrm{macro\ code}\times\mathrm{rescale\ factor}.$$

The command emits one complete `rescale_factors = [...]` assignment, preserving entries for modes not selected by `--modes`, and plots the paired maximum-resolution output codes with the fitted line. Lower ADC resolutions derive their effective factors from the macro family law.

## Comparing against an idealized macro

`--cim_macro ideal` swaps the configured macro for its `to_ideal()` twin while retaining its calibrated factors and integer ADC resolution. `--cim_macro physical` runs the macro as configured. The twin removes circuit nonidealities but does not bypass quantization unless called explicitly with `adc_active_bits = 0`, a value accepted only by `IdealCimMacro`.

A standalone ideal config is another option. It instantiates an ideal macro from hand-authored parameters tied to no fabricated chip and is suitable for flow bring-up rather than hardware validation.
