# Internals

How the NeuroX codebase is built and why.

Cross-cutting:

- [config_and_policy](config_and_policy.md) — frozen configs, `from_config` dispatch, owner-constructs-child
- [physical_state](physical_state.md) — physical state at nominal / actual / snap tiers, evolved by the `__init__` / `fabricate` / `program` / `snapshot` lifecycle
- [package_surface](package_surface.md) — package imports, exports, registry import completeness, and re-export exceptions
- [compile contracts](compile/contracts.md) — the dynamo-safety requirements
- [regional compilation](compile/scheme_a_regional.md) — the active solver compilation scheme

Use the site navigation to browse subsystem bases and concrete implementations.
