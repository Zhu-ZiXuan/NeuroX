# `neurox/analog/tia/base.py`

## Current role

`TIA` is the abstract base for the transimpedance-amp clamp-driver family. It carries:

- the per-family `config_type → impl_class` registry (via `RegistryMixin[type[TIAConfig], TIA]`)
- the family-level `from_config(...)` classmethod
- profiler registration
- the abstract `v_ref__V` property every concrete TIA must implement
- the snapshot type parameter (`Generic[SnapshotT]`, bound to `TIASnapshot`): each concrete TIA fixes its own snapshot dataclass, and `snapshot()` / `solve_clamp()` carry that concrete type

`TIAConfig` is the family base config: only the orchestration-level fields every TIA topology shares (`v_ref__V`, leakage / area / latency). Concrete TIA configs inherit and add their topology-specific design parameters.

`TIAPolicy` is the empty marker base policy for the family. Concrete TIA implementations declare their own concrete `*Policy(TIAPolicy)` carrying that topology's switches (e.g. `OpAmpTIAPolicy`); the composite that holds a TIA stores the abstract `TIAPolicy` field type and the caller passes the concrete impl.

## Family-wide init signature

Every concrete TIA impl exposes the same explicit signature:

```
__init__(self, *, config, policy, name, inst_shape, dtype, T__K)
```

All six arguments are required, keyword-only, and may not be `None`. The base stores `self._inst_shape`, registers profiler bookkeeping, and accepts/discards `config / policy / dtype / T__K` so the dispatcher type-checks cleanly; concrete subclasses store the rest on `self`. The family inherits `FabricateMixin` — concrete TIAs override `_sample_fabricate_mismatch`.

No family-specific runtime extras.

## Abstract `v_ref__V`

`v_ref__V` is declared as an `@property @abstractmethod` on the base. The base owns no `config` storage so it cannot implement the accessor itself. Every concrete TIA implements it from its own `self.config` — typically as `return self.config.v_ref__V`. This is the only solver-facing contract every TIA topology must honour explicitly.

## Solver-facing surface

The 1T1R solver binds `bl_driver: TIA` directly as the BL boundary actor. The contract exposed to the solver is `solve_clamp(i_port__uA, snapshot, *, v_clamp_init__V) -> (v_clamp__V, dVclamp_dI__MOhm)` plus the `v_ref__V` property — nothing else. There is no shared abstract base across TIA and `Driver`; the solver knows it accepts a `TIA` on BL and a `Driver` on SL because that is the 1T1R topology fact.

See also:

- `opamp_tia.md`
- `docs/modules/common/registry_dispatch.md`
