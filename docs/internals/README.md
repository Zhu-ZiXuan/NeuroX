# Internals

How the NeuroX codebase is built and why.

Cross-cutting:

- [config_and_policy](config_and_policy.md) — frozen configs, `from_config` dispatch, owner-constructs-child
- [physical_state](physical_state.md) — physical state at nominal / actual / snap tiers, evolved by the `__init__` / `fabricate` / `program` / `snapshot` lifecycle
- [package_surface](package_surface.md) — package imports, exports, registry import completeness, and re-export exceptions
- [compile](compile/README.md) — where `@torch.compile` applies, the dynamo-safety contracts, and the regional-compilation scheme

Shared primitives (mirrors `neurox/common/`):

- [common](common/README.md) — the software primitives shared across every subsystem

Per-subsystem (mirrors Reference):

- [device](primitive/device/README.md)
- [analog](primitive/analog/README.md)
- [digital](primitive/digital/README.md)
- [xbar](primitive/xbar/README.md)
- [unit](architecture/unit/README.md)
