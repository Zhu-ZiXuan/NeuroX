# Internals

How the NeuroX codebase is built and why.

Cross-cutting:

- [config_and_policy](config_and_policy.md) — frozen configs, `from_config` dispatch, owner-constructs-child
- [physical_state](physical_state.md) — physical state at nominal / actual / snap tiers, evolved by the `__init__` / `fabricate` / `program` / `snapshot` lifecycle
- [package_surface](package_surface.md) — package imports, exports, registry import completeness, and re-export exceptions
- [compile contracts](compile/contracts.md) — the dynamo-safety requirements
- [regional compilation](compile/scheme_a_regional.md) — the active solver compilation scheme
- [encoding](common/encoding/encodings.md) — integer and signed-digit transcoding
- [prober](common/prober.md) — full-fidelity diagnostic capture links
- [profiler](common/profiler.md) — PPA record collection and the names records carry
- [quantization](common/quant.md) — shared quantization kernels
- [recorder](common/recorder.md) — the shared side channel every record family collects through
- [reporter](common/reporter.md) — records into rows: name resolution, aggregation, and the canonical dump
- [tensor groups](common/tensor_group.md) — dataclass-of-tensors traversal and shape surface

Use the site navigation to browse subsystem bases and concrete implementations.
