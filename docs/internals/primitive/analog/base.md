# Analog base

`AnalogBase` is the root every analog circuit block inherits. The family shares only a config / policy type pair and a construction convention; each block's value path, parameters, and static-PPA declaration are its own. The concrete members are indexed in the [README](README.md).

## Design decisions

- **Empty markers `AnalogConfig` / `AnalogPolicy`.** The family shares no field across its leaves, so each carries its own `*Config` / `*Policy` subclass; the markers exist only to give the family a named config / policy base type.
- **Per-instance PPA data is declared where it is carried.** The `AnalogConfig` marker declares no PPA field, so the `area_per_inst__um2` / `leakage_per_inst__uW` pair sits on a standalone block's own config, or once on a polymorphic family's config base for every member. Whether a block carries the pair at all is its own accounting call rather than a family property: a block whose silicon an owning circuit already budgets declares neither field and sets `is_profile_target` false, which excludes it from the profiler's static walk and forbids it from emitting a dynamic event of its own.

## Contracts & invariants

- **Leaf construction.** Every concrete analog block's `__init__` takes `dtype` and the operating temperature `T__K` on top of the [`ModuleBase`](../../config_and_policy.md) arguments it forwards up.

---

- **Reference**: N/A — software base; per-leaf area / leakage numbers are specified under [reference](../../../reference/primitive/analog/README.md)
- **Implementation**: `neurox/primitive/analog/base.py`
- **Tests**: TODO — covered indirectly via the profiler's static collection
