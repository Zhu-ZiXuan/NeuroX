# `neurox/xbar/readout/offset_switchcap_mux_adc.py`

## Current role

`OffsetSwitchCapMuxAdcReadOut` is the offset-coded readout chain:

- data-side switch-cap accumulation;
- reference-side switch-cap sampling;
- analog mux transport;
- differential ADC conversion.

It is one concrete `ReadOut` family implementation registered via `ReadOut.register_key(OffsetSwitchCapMuxAdcReadOutConfig)`.

## Ownership and construction

`OffsetSwitchCapMuxAdcReadOutConfig` carries:

- `data_switchcap_config`
- `ref_switchcap_config`
- `analog_mux_config`
- `adc_config`

`OffsetSwitchCapMuxAdcReadOutPolicy(ReadOutPolicy)` is a structured composite policy with one sub-policy per child:

- `data_switchcap: SwitchCapPolicy`
- `ref_switchcap: SwitchCapPolicy`
- `analog_mux: AnalogMuxPolicy`
- `bl_adc: ADCPolicy` — abstract base; the concrete impl (e.g. `GeneralADCPolicy`, `McsSarAdcPolicy`) is passed by the caller.

The readout constructs its children itself, driven by the family-wide init bundle (`data_num`, `digit_weights`):

- `data_switchcap = SwitchCap(config=data_switchcap_config, policy=policy.data_switchcap, …, cap_weights=digit_weights)` — one cap per digit, weighted by the positional weights.
- `ref_switchcap = SwitchCap(config=ref_switchcap_config, policy=policy.ref_switchcap, …, cap_weights=(1.0,))` — single unit cap, no positional weighting.
- `analog_mux = AnalogMux(config=analog_mux_config, policy=policy.analog_mux, …)` — instantiated directly.
- `bl_adc = ADC.from_config(config=adc_config, policy=policy.bl_adc, …)` — dispatched through the ADC registry.

`data_num` is stored on the instance and consumed at `__init__` to size the data-leg bank's per-instance axis (`inst_shape = (*readout_inst_shape, data_num)`). No external submodule factory closures are part of the current design.

## Grouped lattice

The readout consumes already-grouped voltages:

- `v_data_grouped: (*runtime, group_num, data_num, digit_num)`
- `v_ref_grouped: (*runtime, group_num)`

The ref leg is expanded across the per-group data axis directly; the readout does not perform a second logical lookup or geometric weighting outside the leaf switch-cap banks.

## Construction-time sub-shape derivation

The readout owns `inst_shape = (*prefix, group_num)`. At `__init__` it derives each child's `inst_shape`:

- `data_switchcap.inst_shape = (*prefix, group_num, data_num)`
- `ref_switchcap.inst_shape = (*prefix, group_num)`
- `analog_mux.inst_shape = (*prefix, group_num, 1)`
- `bl_adc.inst_shape = (*prefix, group_num, 1)`

These shapes are committed when the children are constructed. `fabricate()` is the inherited `FabricateMixin` auto-cascade — no per-call shape arguments anywhere.
