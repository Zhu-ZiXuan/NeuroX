# LinearCimUnit

The engine-backed CIM unit exposing the linear operator: `LinearUnit` × `EngineBackedCimUnit`.

## Design decisions

- **Operator by inheritance, substrate by delegation.** `linear` is the concrete `LinearUnit` template (size-1 `M` plane axis over the engine-delegated `_matmul`, plus the int64 bias add); the unit adds only the `program(weight, bias=None)` mapping — the weight through the `_weight_to_matrix` seam (identity for linear) to `engine.program`, the bias to `_program_int_bias` with `channels = N`.
- **No fields beyond the inherited set.** `LinearCimUnitConfig` / `LinearCimUnitPolicy` add nothing; the engine variant is the nested config's choice.
- **MRO.** `LinearCimUnit(LinearUnit, EngineBackedCimUnit)` linearizes cleanly because `CimUnit` already inherits `UnitBase`; the constructor calls `_init_int_bias_slot` after the engine build.

## Contracts & invariants

- Registered via `@CimUnit.register_key(LinearCimUnitConfig)`.
- `linear` accepts any leading dim set over the trailing `[K]` — every leading dim (including a caller time axis) is a broadcast batch dim that rides through untouched.
- The engine-delegated `_matmul` never includes the bias; only `linear` adds it.

---

- **Reference**: [unit family](../../../../reference/architecture/unit/family.md), [engine family](../../../../reference/architecture/unit/cim/engine/README.md)
- **Implementation**: `neurox/architecture/unit/cim/linear.py`
- **Tests**: `tests/architecture/unit/test_operator_cim_unit.py`
