# `neurox/device/wire.py`

## Current role

`Wire` models one interconnect object with:

- process/spec information that belongs to the wire model itself
- fabricated per-segment state derived from explicit segment resistance input

## Resistance vs. capacitance split

Current project rule:

- wire capacitance density (`c__fF_per_um`) lives in `WireConfig`
- per-segment resistance is a **fabricate-time input tensor**

The wire object does not infer segment resistance from geometry by itself. Owning circuits build the concrete segment resistance tensors and hand them to `Wire.fabricate(...)`.

This keeps:

- the wire model local and reusable
- geometry ownership in the consuming circuit / array
- the wire config free of runtime shape-specific information

## Construction

`Wire.__init__(*, cfg, T__K, dtype)` — no `name` parameter. Devices do not own a profiler name; wire energy / area rolls up to the consuming circuit.
