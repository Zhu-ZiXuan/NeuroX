# `neurox/analog/dac/base.py`

## Current role

`DAC` is the abstract base for the DAC family. It carries only the system-level scaffolding — the per-family registry inherited from `RegistryMixin[type[DACConfig], DAC]`, profiler registration, and the family-level `from_config(...)` classmethod.

`DACConfig` is the empty marker config used as the polymorphic-field type on parent configs.

## Family-wide init signature

Every concrete `DAC` impl exposes the same explicit signature:

```
__init__(self, *, cfg, name, inst_shape, dtype, T__K)
```

All five arguments are required, keyword-only, and may not be `None`. The base `__init__` accepts the same signature so the dispatcher in `from_config` type-checks cleanly; it stores `self._inst_shape` and uses `name` for profiler registration. `cfg` / `dtype` / `T__K` stay on the concrete subclass (see [`docs/dev/architecture/state_holding.md`](docs/dev/architecture/state_holding.md)).

## Required subclass surface

`DAC` inherits `FabricateMixin`. The auto-cascade `fabricate()` is provided by the mixin; subclasses override `_sample_fabricate_mismatch` only if they introduce static per-output mismatch. The default body is a no-op.

Concrete subclasses must implement:

- `convert(code)` — map integer codes to analog voltages.
- `code_to_signal` — the code → nominal-voltage LUT (a tensor of length `n_codes`). Read directly to obtain a nominal operating-point voltage without going through `convert`.
- Cost-metric properties: `area_per_inst__um2`, `leakage_per_inst__uW`, `latency_per_op__ns`.

See also:

- `docs/dev/architecture/config_and_construction.md`
- `docs/dev/modules/common/registry_dispatch.md`
