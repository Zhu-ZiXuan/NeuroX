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

The readout constructs its children itself, driven by the family-wide init bundle (`data_num`, `digit_weights`):

- `data_switchcap = SwitchCap(cfg=data_switchcap_cfg, …, cap_weights=digit_weights)` — one cap per digit, weighted by the positional weights.
- `ref_switchcap = SwitchCap(cfg=ref_switchcap_cfg, …, cap_weights=(1.0,))` — single unit cap, no positional weighting.
- `analog_mux = AnalogMux(cfg=analog_mux_cfg, …)` — instantiated directly.
- `bl_adc = ADC.from_config(cfg=adc_cfg, …)` — dispatched through the ADC registry.

`data_num` is stored on the instance and consumed by `fabricate(shape)` to size the data-leg bank's per-instance axis. No external submodule factory closures are part of the current design.

## Grouped lattice

The readout consumes already-grouped voltages:

- `v_data_grouped: (*runtime, group_num, data_num, digit_num)`
- `v_ref_grouped: (*runtime, group_num)`

The ref leg is expanded across the per-group data axis directly; the readout does not perform a second logical lookup or geometric weighting outside the leaf switch-cap banks.

## Fabricate signature

`fabricate(shape)` takes only the readout's own virtual-instance shape `(*prefix, group_num)`. Internally it dispatches:

- `data_switchcap.fabricate((*prefix, group_num, data_num))`
- `ref_switchcap.fabricate((*prefix, group_num))`
- `analog_mux.fabricate((*prefix, group_num, 1))`
- `bl_adc.fabricate((*prefix, group_num, 1))`
