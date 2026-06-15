# `neurox/xbar/readout/base.py`

## Current role

`ReadOut` is the abstract base for the readout-chain family. It bundles every voltage-domain block between an array's per-column boundary output and the final ADC code into one fabricate / readout lifecycle.

`ReadOutConfig` is the family base config carrying only orchestration-level energy / PPA knobs. Concrete readout configs inherit and add their topology-specific member configs (data / ref switchcap, mux, ADC).

`ReadOutPolicy` is the empty marker base policy for the family. Concrete readout impls declare their own structured `*Policy(ReadOutPolicy)` (e.g. `OffsetSwitchCapMuxAdcReadOutPolicy`) carrying nested sub-policies for every child the impl owns. The composite that holds a ReadOut stores the abstract `ReadOutPolicy` field type and the caller passes the concrete impl.

## Family-wide init signature

Every concrete readout impl exposes:

```
__init__(self, *, config, policy, name, inst_shape, dtype, T__K, data_num, digit_weights)
```

- `config / policy / name / inst_shape / dtype / T__K` — standard leaf-init bundle. `inst_shape` is `(*prefix, group_num)`.
- `data_num: int` — number of data per reference group; fixes the data-leg SwitchCap's bank-axis size.
- `digit_weights: tuple[float, ...]` — per-digit positional weights (length `digit_num`); drives the data-leg SwitchCap's `cap_weights`.

`data_num` and `digit_weights` are structural facts of the consuming xbar / macro and are committed at construction. `digit_weights` is a Python tuple — tensor construction happens inside the leaf SwitchCap.

`ReadOut.from_config(...)` forwards all of these through to the registered impl. The family inherits `FabricateMixin`; sub-modules (data/ref switchcaps, mux, ADC) are constructed with the appropriate derived `inst_shape` and cascade automatically.

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
- `data_num` — number of data per group (init-time constant).
- `digit_num` — number of digits per data (init-time constant; equals `len(digit_weights)`).

The readout's own instance shape `(*prefix, group_num)` is committed via `inst_shape` at `__init__`; `fabricate()` is the inherited auto-cascade trigger and takes no shape argument.

`readout(v_data_grouped, v_ref_grouped, *, adc_operation_point)` is the per-VMM kernel; it returns the per-data ADC code.

## `readout(...)` vs `forward(...)`

The readout is a non-trainable analog chain and never participates in autograd. Exposing the per-VMM kernel as `readout(...)` rather than `__call__` / `forward` skips the `nn.Module` hook plumbing and keeps the call site explicit.

See also:

- `offset_switchcap_mux_adc.md`
- `docs/modules/common/registry_dispatch.md`
