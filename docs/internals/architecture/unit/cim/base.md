# CimUnit base

The `CimUnit` registry root — the config / policy roots (the config carries the unit-local peripheral PPA) and the family factory — plus the unregistered `EngineBackedCimUnit` intermediate that owns a `CimEngine` and delegates the execution surface to it.

## Design decisions

- **Registry dispatch keyed on config and policy types.** A member declares `@CimUnit.register_neurox_module(config_type=..., policy_type=...)`; `from_config` passes both objects to `_lookup_neurox_module`, and `RegistryMixin` constructs the internal type-pair key. A mismatched pair fails before construction.
- **The lowering machinery is inherited, not redeclared.** `CimUnit` mixes in `UnitBase`, so the abstract value-range / ADC surface, the protected `_matmul` seam, and the lowering template flow from the [unit base](../base.md); the operator surface (`linear` / `conv2d`, `program`) comes from the `UnitBase`-derived operator ABC each concrete leaf mixes in. The class adds only the module-tree construction context (`w_logical_shape`, `dtype`, `T__K`, `ideal_macro`) and the registry factory.
- **The base owns no `xbar` and no pipeline.** Slicing, tiling, macro cycling, and aggregation live in the [engine base](engine/base.md); the ideal leaves own no engine at all. The root therefore carries no execution scaffolding.
- **`EngineBackedCimUnit` owns an engine.** The intermediate builds `CimEngine.from_config(config.engine, ...)` and delegates `_matmul`, both value ranges, and the ADC surface to `self.engine`. Operator-specific mapping remains outside the intermediate.
- **Config and policy trees have the same ownership boundary.** `EngineBackedCimUnitConfig.engine` holds the concrete engine config and `EngineBackedCimUnitPolicy.engine` holds the corresponding engine policy. The unit passes both objects unchanged to `CimEngine.from_config`; policy TOMLs therefore use the same engine nesting as config TOMLs.
- **Unit-local peripheral PPA only.** The unit's reported area and leakage cover **only** its own peripheral overhead: the engine's tile and reducers self-report their own silicon, so the unit's numbers deliberately exclude them and nothing is double-counted. The values are a TODO data-gap — no real per-unit number is available yet, so construction sites pass `0.0`.

## Contracts & invariants

- **Uniform construction.** `from_config` builds every registered member through one call shape (`config`, `policy`, `w_logical_shape`, `dtype`, `T__K`, `ideal_macro`), so each member must accept the base's construction arguments unchanged — narrowing or reordering them breaks dispatch. `w_logical_shape` must have at least two trailing dims (a member may constrain it further, e.g. conv2d's 4-D kernel shape).
- **Config carries the unit-local PPA; policy root is empty.** `CimUnitConfig` carries and validates the two required peripheral-PPA fields `area_per_inst__um2` / `leakage_per_inst__uW`; its concrete type and the concrete `CimUnitPolicy` type form the registry key. `EngineBackedCimUnitConfig` adds `engine: CimEngineConfig`; `EngineBackedCimUnitPolicy` adds the parallel `engine: CimEnginePolicy` dispatch point.
- **The unit is a fabricate container.** `_sample_fabricate_mismatch` is a pass; `fabricate()` cascades through the owned engine (itself a `ModuleBase` container) down to the tile and reducers.
- **The CIM-backed boundary enforces integer activations.** `EngineBackedCimUnit._matmul` rejects floating, complex, and boolean tensors before delegating to `engine.matmul`; the generic operator bases and ideal references do not impose that runtime gate.

---

- **Reference**: [unit family](../../../../reference/architecture/unit/family.md)
- **Implementation**: `neurox/architecture/unit/cim/base.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`
