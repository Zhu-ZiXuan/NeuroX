# Internals

How the NeuroX codebase is built and why — the implementation companion to [Reference](../reference/README.md). Cross-cutting documents cover global patterns with no single code home; `common/` mirrors the shared-primitive package; per-subsystem documents mirror Reference.

Cross-cutting:

- [config_and_construction](config_and_construction.md) — frozen configs, `from_config` dispatch, owner-constructs-child
- [state_holding](state_holding.md) — the nominal / actual / snapshot state model
- [fabrication_lifecycle](fabrication_lifecycle.md) — `__init__` / `fabricate` / `program` / `snapshot` lifecycle
- [compile](compile/README.md) — where `@torch.compile` applies, the dynamo-safety contracts, and the regional-compilation scheme (current and deferred)

Shared primitives (mirrors `neurox/common/`):

- [common](common/README.md) — `CircuitBase`, config load/dump, nonideality kernels, physical constants, quantization, the profiler, and the reusable mixins

Per-subsystem (mirrors Reference):

- [device](device/README.md)
- [analog](analog/README.md)
- [digital](digital/README.md)
- [xbar](xbar/README.md)
- [macro](macro/README.md)
