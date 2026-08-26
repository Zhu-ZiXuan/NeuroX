# Macro calibration

Macro calibration owns the mapping between network-level numerical ranges and a complete macro's output codes. It is independent of the ADC's analog threshold placement.

## Derive the mode set

`mode_derive` reads one inclusive numerical range per layer, maps each range to a canonical unsigned or mid-zero integer window, clusters compatible windows, and emits the macro quantization modes plus the layer-to-mode map.

```bash
python -m neurox.tools.calibrate_macro.mode_derive \
    --config validations/<paper>/tools/<mode_derive_run>.toml \
    --output validations/<paper>/tools/modes.toml
```

A design with one fixed operating mode may write that small mode-set TOML directly.

## Determine the output rescale factor

When a macro declares an exact integer mapping from ideal MAC units to output codes, derive the rescale factor from that mapping and store it directly. Do not fit a quantity whose numerical meaning is already fixed by design. Use `rescale_fit` only when the complete macro transfer leaves that relationship empirical: it compares the public output of a physical macro with the public lossless output of its ideal twin, without inspecting the ADC type or using an ADC probe. Logical weights have shape `[Bw, input_num, output_num]`; logical inputs have shape `[Bx, 1, input_num]` and are split into legal `max_active_num` planes before both macros run.

```bash
python -m neurox.tools.calibrate_macro.rescale_fit \
    --config validations/<paper>/tools/calibrate_macro_rescale.toml \
    --device cuda:0
```

The zero-through-origin fit estimates `ideal_value = max_bits_rescale_factor * macro_code` at `adc_max_bits`. Lower resolutions follow the macro family's bit-width law. The command emits one `[[modes]]` fragment per selected mode and plots the paired output codes with the fitted line.
