# `neurox/analog/tia/base.py`

## Current role

`TIA` is the abstract base for the transimpedance-amp clamp-driver family. It carries:

- the per-family `config_type → impl_class` registry (via `RegistryMixin[type[TIAConfig], TIA]`)
- the family-level `from_config(...)` classmethod
- profiler registration
- the abstract `v_ref__V` property every concrete TIA must implement

`TIAConfig` is the family base config: only the orchestration-level fields every TIA topology shares (`v_ref__V`, leakage / area / latency). Concrete TIA configs inherit and add their topology-specific design parameters.

## Family-wide init signature

Every concrete TIA impl exposes the same explicit signature:

```
__init__(self, *, cfg, name, inst_shape, dtype, T__K)
```

All five arguments are required, keyword-only, and may not be `None`. The base stores `self._inst_shape`, registers profiler bookkeeping, and accepts/discards `cfg / dtype / T__K` so the dispatcher type-checks cleanly; concrete subclasses store the rest on `self`. The family inherits `FabricateMixin` — concrete TIAs override `_sample_fabricate_mismatch`.

No family-specific runtime extras.

## Abstract `v_ref__V`

`v_ref__V` is declared as an `@property @abstractmethod` on the base. The base owns no `cfg` storage so it cannot implement the accessor itself. Every concrete TIA implements it from its own `self.cfg` — typically as `return self.cfg.v_ref__V`. This is the only solver-facing contract every TIA topology must honour explicitly.

## ClampDriver protocol

A TIA is one concrete implementer of the [`ClampDriver`](docs/dev/modules/analog/clamp_driver.md) protocol. Solvers consume the clamp boundary structurally through that protocol — they do not import `TIA` directly.

See also:

- `opamp_tia.md`
- `docs/dev/modules/analog/clamp_driver.md`
- `docs/dev/modules/common/registry_dispatch.md`
