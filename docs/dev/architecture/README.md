# Architecture Docs

This directory stores the project-wide rules that cut across modules. Every architecture document is **the** source of truth for one concern; module docs and source comments point here rather than restating.

## Three documentation layers

NeuroX keeps three independent layers of written documentation:

- **Docstring** — short, local, describes the current interface of one symbol.
- **Inline comment** — local implementation help for the block of code it sits next to.
- **`docs/dev/`** — long-lived design rules, conventions, and module-level descriptions.

These layers must stay separate; the same rule is never restated across all three. See [`code_style.md`](code_style.md).

## Documents

### Conventions

- [`code_style.md`](code_style.md) — docstring, inline-comment, shape-annotation, and dependency-direction rules.
- [`naming_conventions.md`](naming_conventions.md) — physical-quantity suffixes, dataclass role suffixes (`*Config` / `*Snapshot` / `*DCOP` / `*Plan` / `*Result`), primary-method names, hierarchical profiler names.
- [`physical_units.md`](physical_units.md) — tensor units, config-side units, suffix grammar, dtype policy on the runtime path.

### Lifecycle and state

- [`config_and_construction.md`](config_and_construction.md) — process / design / spec / runtime parameter classes, device-vs-circuit config split, family-base + concrete-config pattern, explicit `from_config(...)`, and the paired `*Policy` runtime kwarg (no field defaults; loaded from a separate policy TOML or constructed in code).
- [`fabrication_lifecycle.md`](fabrication_lifecycle.md) — the `__init__` / `fabricate` / `snapshot` / `forward` lifecycle; shape is committed at `__init__`, `fabricate()` is the no-arg auto-cascade, re-callable via nominal templates.
- [`state_holding.md`](state_holding.md) — three-stage state model (nominal → actual → snapshot), ownership rule, snapshot pattern.

### Runtime

- [`compile_policy.md`](compile_policy.md) — where `@torch.compile` is applied (macro entry only), forbidden behaviours on the compiled path, intentional graph breaks (structural `_log_dynamic_energy` / `_log_latency` + temporary `Offset1T1RXbar.vec_mat_mul`).
- [`profiler_and_ppa.md`](profiler_and_ppa.md) — `CircuitBase` PPA surface, `ProfileMixin` side-channel, `_log_dynamic_energy` / `_log_latency` independent entries, composite-module aggregation, profiler context manager + batched sync.

### Mapping

- [`mapping.md`](mapping.md) — macro / mapper / tiler / slicer / transcoder / xbar layering.

### Task index

- [`recipes.md`](recipes.md) — per-task checklists (add a device, add a leaf circuit, add a family member, add a new family, …). Pure index into the documents above; no prose duplication.

## Relationship with other directories

- `docs/dev/modules/` mirrors `neurox/`; each Python module gets its own design note describing its current responsibility and protocol. Module docs cite architecture documents rather than restating them.
- `docs/dev/adr/` records architectural decisions and their rejected alternatives. ADRs explain **why**; architecture documents describe **what is current**.
