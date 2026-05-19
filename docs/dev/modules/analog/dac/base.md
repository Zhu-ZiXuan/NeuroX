# `neurox/analog/dac/base.py`

## Current role

`DAC` is the abstract base for the DAC family. It carries only the system-level scaffolding — the per-family registry inherited from `ConfigDispatchMixin[DACConfig, DAC]`, profiler registration, and the family-level `from_config(...)` classmethod.

`DACConfig` is the empty marker config used as the polymorphic-field type on parent configs.

## Family-wide init signature

Every concrete `DAC` impl exposes the same explicit signature:

```
__init__(self, *, cfg, name, T__K, dtype)
```

All four arguments are required, keyword-only, and may not be `None`. The base `__init__` accepts the same signature so the dispatcher in `from_config` type-checks cleanly, but it only consumes `name` for profiler registration — `cfg` / `T__K` / `dtype` stay on the concrete subclass (see [`architecture/state_holding.md`](../../../architecture/state_holding.md)).

## Required subclass surface

Concrete subclasses must implement:

- `convert(code)` — map integer codes to analog voltages.
- `code_to_signal` — the code → nominal-voltage LUT (a tensor of length `n_codes`). Read directly to obtain a nominal operating-point voltage without going through `convert`.
- Cost-metric properties: `area_per_inst__um2`, `leakage_per_inst__uW`, `latency_per_op__ns`.

`fabricate(shape)` is a default no-op on the base; subclasses override only if they introduce static per-output mismatch.

See also:

- `../../../architecture/config_and_construction.md`
- `../../common/config_dispatch.md`
