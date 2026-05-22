# `neurox/analog/analog_mux.py`

`AnalogMux` is a leaf analog transport block.

It owns:

- its own design/spec config
- per-call differential transport behaviour and energy

`AnalogMux` inherits `FabricateMixin`, but it has no static mismatch of its own — all of its noise (CM / DM) is dynamic and applied inside `transport`. The mixin's default `_sample_fabricate_mismatch` no-op applies.

## Construction

`__init__(*, cfg, name, inst_shape, dtype, T__K)`. The per-instance count is locked from `inst_shape` at construction.

It is not polymorphic today, so parent circuits construct it directly from its config rather than dispatching through a family base.
