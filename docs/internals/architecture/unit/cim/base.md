# CimUnit base

The `CimUnit` registry root — the config / policy roots (the config carries the unit-local peripheral PPA) and the family factory — plus the unregistered `EngineBackedCimUnit` intermediate that owns a `CimEngine` and delegates the execution surface to it.

## Design decisions

- **Registry dispatch keyed on config type.** `from_config` looks up the impl registered for `type(config)` via `RegistryMixin`; adding a member is one registration and never changes the base.
- **The lowering machinery is inherited, not redeclared.** `CimUnit` mixes in `UnitBase`, so the abstract value-range / ADC surface, the protected `_matmul` seam, and the lowering template flow from the [unit base](../base.md); the operator surface (`linear` / `conv2d`, `program`) comes from the `UnitBase`-derived operator ABC each concrete leaf mixes in. The class adds only the module-tree construction context (`w_logical_shape`, `dtype`, `T__K`, `ideal_xbar`) and the registry factory.
- **The base owns no `xbar` and no pipeline.** Slicing, tiling, macro cycling, and aggregation live in the [engine base](engine/base.md); the ideal leaves own no engine at all. The root therefore carries no execution scaffolding.
- **`EngineBackedCimUnit` owns an engine.** The intermediate builds `CimEngine.from_config(config.engine, ...)` and delegates `_matmul`, both value ranges, and the ADC surface to `self.engine`. Operator-specific mapping remains outside the intermediate.
- **Config and policy trees have the same ownership boundary.** `EngineBackedCimUnitConfig.engine` holds the concrete engine config and `EngineBackedCimUnitPolicy.engine` holds the corresponding engine policy. The unit passes both objects unchanged to `CimEngine.from_config`; policy TOMLs therefore use the same engine nesting as config TOMLs.
- **Unit-local peripheral PPA only.** The unit's reported area and leakage cover **only** its own peripheral overhead: the engine's tile and reducers self-report their own silicon, so the unit's numbers deliberately exclude them and nothing is double-counted. The values are a TODO data-gap — no real per-unit number is available yet, so construction sites pass `0.0`.

## Contracts & invariants

- **Uniform construction.** `from_config` builds every registered member through one call shape (`config`, `policy`, `w_logical_shape`, `dtype`, `T__K`, `ideal_xbar`), so each member must accept the base's construction arguments unchanged — narrowing or reordering them breaks dispatch. `w_logical_shape` must have at least two trailing dims (a member may constrain it further, e.g. conv2d's 4-D kernel shape).
- **Config carries the unit-local PPA; policy root is empty.** `CimUnitConfig` carries the two required peripheral-PPA fields `area_per_inst__um2` / `leakage_per_inst__uW` (wired through `validate_ppa`) and is the registry key; `CimUnitPolicy` is an empty marker. `EngineBackedCimUnitConfig` adds `engine: CimEngineConfig`; `EngineBackedCimUnitPolicy` adds the parallel `engine: CimEnginePolicy` dispatch point.
- **The unit is a fabricate container.** `_sample_fabricate_mismatch` is a pass; `fabricate()` cascades through the owned engine (itself a `ModuleBase` container) down to the tile and reducers.
- **Delegation is total on the execution surface.** `EngineBackedCimUnit._matmul` forwards to `engine.matmul` verbatim, as do the published ranges and the ADC surface; operator lowering remains outside this base.

---

- **Reference**: [unit family](../../../../reference/architecture/unit/family.md)
- **Implementation**: `neurox/architecture/unit/cim/base.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`
