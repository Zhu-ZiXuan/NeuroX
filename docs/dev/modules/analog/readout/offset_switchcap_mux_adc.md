# `neurox/analog/readout/offset_switchcap_mux_adc.py`

## Current role

`OffsetSwitchCapMuxAdcReadOut` is the offset-coded readout chain:

- data-side switch-cap accumulation;
- reference-side switch-cap sampling;
- analog mux transport;
- differential ADC conversion.

It is one concrete `ReadOut` family implementation registered via `ReadOut.register_config(OffsetSwitchCapMuxAdcReadOutConfig)`.

## Ownership and construction

`OffsetSwitchCapMuxAdcReadOutConfig` carries:

- `data_switchcap_cfg`
- `ref_switchcap_cfg`
- `analog_mux_cfg`
- `adc_cfg`

The readout constructs these children itself:

- `SwitchCap` and `AnalogMux` are instantiated directly
- `ADC` is dispatched through `ADC.from_config(...)`

No external submodule factory closures are part of the current design.

## Grouped lattice

The readout consumes already-grouped voltages:

- `v_data_grouped: (*runtime, group_num, data_num, digit_num)`
- `v_ref_grouped: (*runtime, group_num)`

The ref leg is expanded across the per-group data axis directly; the readout does not perform a second logical lookup or geometric weighting outside the leaf switch-cap banks.
