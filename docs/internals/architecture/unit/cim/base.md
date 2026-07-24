# CimUnit base

The `CimUnit` registry root — the config / policy roots (the config carries the unit-local peripheral PPA) and the family factory — plus the unregistered `EngineBackedCimUnit` intermediate that owns a `CimEngine` and delegates the execution surface to it.

## Design decisions

- **Registry dispatch keyed on config type.** `from_config` looks up the impl registered for `type(config)` via `RegistryMixin`; adding a member is one `@CimUnit.register_key(MyConfig)` line and never touches the base. The cost — one config dataclass per member — is paid once. The substrate-free ideal leaves join the same registry, so config-driven unit selection is uniform across physical and reference members.
- **The lowering machinery is inherited, not redeclared.** `CimUnit` mixes in `UnitBase`, so the abstract value-range / ADC surface, the protected `_matmul` seam, and the lowering template flow from the [unit base](../base.md); the operator surface (`linear` / `conv2d`, `program`) comes from the `UnitBase`-derived operator ABC each concrete leaf mixes in. The class adds only the module-tree construction context (`w_logical_shape`, `dtype`, `T__K`, `ideal_xbar`) and the registry factory.
- **The base owns no `xbar` and no pipeline.** Slicing, tiling, macro cycling, and aggregation live in the [engine family](engine/README.md); the ideal leaves own no engine at all. The root therefore carries no execution scaffolding.
- **`EngineBackedCimUnit` is composition, not inheritance.** The intermediate builds `CimEngine.from_config(config.engine, ...)` in its constructor — the nested engine config's concrete type selects the variant — and delegates `_matmul`, both value ranges, and the ADC surface to `self.engine`. Concrete subclasses add only their operator's `program` mapping (and, for conv2d, the `_engine_w_logical_shape()` hook that hands the engine a different logical weight shape).
- **Engine policy assembled in code.** `EngineBackedCimUnitPolicy` keeps the `cim_macro_policy` field flat; the unit constructs `CimEnginePolicy(cim_macro_policy=policy.cim_macro_policy)` itself, so policy TOMLs carry no engine nesting.
- **Unit-local peripheral PPA only.** The unit's reported area and leakage cover **only** its own peripheral overhead: the engine's tile and reducers self-report their own silicon, so the unit's numbers deliberately exclude them and nothing is double-counted. The values are a TODO data-gap — no real per-unit number is available yet, so construction sites pass `0.0`.

## Contracts & invariants

- **Uniform construction.** `from_config` builds every registered member through one call shape (`config`, `policy`, `w_logical_shape`, `dtype`, `T__K`, `ideal_xbar`), so each member must accept the base's construction arguments unchanged — narrowing or reordering them breaks dispatch. `w_logical_shape` must have at least two trailing dims (a member may constrain it further, e.g. conv2d's 4-D kernel shape).
- **Config carries the unit-local PPA; policy root is empty.** `CimUnitConfig` carries the two required peripheral-PPA fields `area_per_inst__um2` / `leakage_per_inst__uW` (wired through `validate_ppa`) and is the registry key; `CimUnitPolicy` is an empty marker. `EngineBackedCimUnitConfig` adds the nested `engine: CimEngineConfig` dispatch point; `EngineBackedCimUnitPolicy` adds `cim_macro_policy: CimMacroPolicy`.
- **The unit is a fabricate container.** `_sample_fabricate_mismatch` is a pass; `fabricate()` cascades through the owned engine (itself a `ModuleBase` container) down to the tile and reducers.
- **Delegation is total on the execution surface.** `EngineBackedCimUnit._matmul` forwards to `engine.matmul` verbatim, as do the published ranges and the ADC surface; the unit layer adds no arithmetic beyond the operator lowerings (linear's bias add, conv2d's gather/fold) defined on the operator ABCs.

---

- **Reference**: [reference/architecture/unit/cim](../../../../reference/architecture/unit/cim/README.md)
- **Implementation**: `neurox/architecture/unit/cim/base.py`
- **Tests**: `tests/architecture/unit/test_cim_unit.py`
