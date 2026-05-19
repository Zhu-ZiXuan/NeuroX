# `neurox/analog/readout/base.py`

## Current role

`ReadOut` is the abstract base for the readout-chain family. It bundles every voltage-domain block between an array's per-column boundary output and the final ADC code into one fabricate / readout lifecycle.

`ReadOutConfig` is the family base config carrying only orchestration-level energy / PPA knobs. Concrete readout configs inherit and add their topology-specific member configs (data / ref switchcap, mux, ADC).

## Family-wide init signature

Every concrete readout impl exposes:

```
__init__(self, *, cfg, name, T__K, dtype, stochastic)
```

`stochastic` is the family-specific extra forwarded straight through to the readout's internal ADC. The base accepts and discards `cfg / T__K / dtype / stochastic` for the dispatcher; the concrete subclass stores them.

## Leaf-module-only electrical math

All signal-value changes happen inside the leaf circuit modules (SwitchCap, AnalogMux, ADC). The readout container only performs shape operations (`unflatten`, `unsqueeze`, `expand`) and energy aggregation:

- the readout does not multiply voltages by a "weight" tensor outside a SwitchCap;
- the readout does not re-derive a per-data weighted sum or ref baseline mathematically — those are properties of the SwitchCap kernels;
- the readout does not scale the ref leg by any encoding-specific constant — the per-data negative leg is the ref switchcap output broadcast via `unsqueeze + expand` onto the data lattice.

## Grouped lattice convention

The readout chain operates on a grouped lattice keyed by reference-group structure:

- `*prefix` — physical-instance prefix.
- `*runtime` — runtime tensor prefix (broadcast of `*prefix` with any leading activation batch dims).
- `group_num` — number of reference groups.
- `data_num` — number of data per group.
- `digit_num` — number of digits per data.

`fabricate(shape, *, data_num, digit_weights)` takes the readout container's own virtual-instance shape `(*prefix, group_num)` plus the two independent inputs sizing the data-side bank — the integer `data_num` and the 1-D `digit_weights` template (length `digit_num`).

`readout(v_data_grouped, v_ref_grouped, *, adc_mode, adc_bits)` is the per-VMM kernel; it returns a `ReadOutOutput` with the per-data ADC code plus the post-mux differential voltages.

## `readout(...)` vs `forward(...)`

The readout is a non-trainable analog chain and never participates in autograd. Exposing the per-VMM kernel as `readout(...)` rather than `__call__` / `forward` skips the `nn.Module` hook plumbing and keeps the call site explicit.

See also:

- `offset_switchcap_mux_adc.md`
- `../../common/config_dispatch.md`
