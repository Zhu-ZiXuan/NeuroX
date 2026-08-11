# LinearCimUnit

The engine-backed CIM unit exposing the linear operator: `LinearUnit` × `EngineBackedCimUnit`.

## Design decisions

- **Operator by inheritance, substrate by delegation.** `linear` is the concrete `LinearUnit` template (size-1 `M` plane axis over the engine-delegated `_matmul`, plus the int64 bias add); the unit adds only the `program(weight, bias=None)` mapping — the weight through the `_weight_to_matrix` seam (identity for linear) to `engine.program`, the bias to `_program_int_bias` with `channels = N`.
- **`M` is a structural constant here.** `latency__ns(input_shape, *, adc_bits)` hands the engine `output_plane_num = 1`, mirroring the size-1 unsqueeze `_activation_to_planes` performs, and reads nothing out of `input_shape`: a linear operator lowers to one plane whatever it is handed, so no output-size arithmetic is duplicated.
- **No fields beyond the inherited set.** `LinearCimUnitConfig` / `LinearCimUnitPolicy` add nothing; the engine variant is the nested config's choice.
- **Uniform input blocking is enforced here, not by the macro base.**
  `LinearCimUnitConfig.validate()` requires
  `input_num % max_active_num == 0`. The operator reads every logical input,
  so each phase uses the same calibrated conversion range.
- **MRO.** `LinearCimUnit(LinearUnit, EngineBackedCimUnit)` linearizes cleanly because `CimUnit` already inherits `UnitBase`. Weight programming is delegated to the engine; `_program_int_bias` stores the optional bias as ordinary programmed state.

## Contracts & invariants

- Registered with the `(LinearCimUnitConfig, LinearCimUnitPolicy)` key.
- `linear` accepts any leading dim set over the trailing `[K]` — every leading dim (including a caller time axis) is a broadcast batch dim that rides through untouched.
- The engine-delegated `_matmul` never includes the bias; only `linear` adds it.

---

- **Reference**: [unit family](../../../../reference/architecture/unit/family.md), [engine family](../../../../reference/architecture/unit/cim/engine/family.md)
- **Implementation**: `neurox/architecture/unit/cim/linear.py`
- **Tests**: `tests/architecture/unit/test_linear_cim_unit.py`
