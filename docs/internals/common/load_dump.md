# Load / Dump — Implementation

## Summary

`load_dump` is the sole format boundary between NeuroX's frozen `*Config` dataclass tree and plain TOML / YAML files: no other module knows about file formats, and every config-loading entry point routes through it. It offers three layers of access — in-memory dataclass coercion (`dataclass_from_dict` / `dataclass_to_dict`), file round-trips with directive expansion (`dataclass_from_file` / `dataclass_to_file`, plus the multi-config `dict_configs_from_file` / `dict_configs_to_file`), and raw uncoerced dict access (`dict_from_file` / `dict_to_file`) — over shared primitives `merge_dicts`, `resolve_uses`, and `preset_path`. This is cross-cutting software architecture; it carries no physics and has no reference twin. The physics it serializes is specified under the per-subsystem reference docs.

## Design decisions

- **Sole format boundary.** All TOML / YAML knowledge is concentrated here so that adding a new file format, or changing a parser, touches one module. Callers that want raw data still go through `dict_from_file`; callers that want a built config go through `dataclass_from_file`. The two never re-implement parsing.
- **Three NeuroX-private keys are plain strings.** `_neurox_type`, `_neurox_use`, and `_neurox_use_preset` are string-valued keys that external TOML / YAML parsers treat as ordinary entries. This keeps config files loadable by any standard parser and confines all NeuroX semantics to this module's resolution passes rather than to a custom file dialect.
- **Coercion is family-agnostic.** `dataclass_from_dict` walks `typing.get_type_hints(cls)` generically and never enumerates which polymorphic families exist; `_neurox_type` is resolved by searching `cls`'s dataclass descendants for a matching `__name__`. Adding a config family therefore needs no loader change.
- **Unknown keys are an error, not a silent default.** A key not matching any dataclass field raises `TypeError` listing the valid fields. The alternative — dropping unrecognized keys — would let a misspelled parameter silently fall back to its default and ship a wrong number. Rejected in favor of fail-loud.
- **Two directive flavors, split by path-resolution base.** `_neurox_use` resolves relative to the file that carries it (user-side sibling fragments, versioned with the user's project); `_neurox_use_preset` resolves relative to `neurox/presets/` (library reference parameters, versioned with the `neurox` package). Splitting them keeps a moved or copied preset from silently re-binding its references against a user directory.
- **Presets are a closed dependency graph.** A subtree reached through `_neurox_use_preset` is in *preset mode*: encountering `_neurox_use` anywhere inside it raises `ValueError`. Every preset's references therefore stay inside the package, so relocating a preset file cannot re-bind it to a user-side fragment.
- **Inline keys override the fragment, not the reverse.** A directive supplies defaults; the inline keys at the call site customise. This makes a `_neurox_use` reference behave like "start from this fragment, then patch", which is the common config-authoring idiom.
- **Preset paths are forward-only.** A preset reference rejects leading `./`, any `..` segment, and absolute paths. Presets are addressed only by a forward relative path from the presets root, so a reference cannot escape the package tree.
- **`preset_path(rel)` for direct library loads; the directive for user references.** Library code that loads a bundled preset itself uses `preset_path` and `dataclass_from_file`; a user file that wants to pull in a bundled fragment uses `_neurox_use_preset`. The two entry points share suffix inference and path validation but serve different callers.

## Contracts & invariants

- **In-memory round-trip.** `dataclass_from_dict(cls, data)` and `dataclass_to_dict(obj)` are the coercion pair with no file I/O. `dataclass_to_dict` emits a nested dict ready for dumping; `Enum` values are written as their `.value`, and a `_neurox_type` discriminator is emitted only for dataclasses that participate in a polymorphic family (a dataclass ancestor or descendant exists), keeping standalone round-trips terse.
- **Recursive coercion contract.** `dataclass_from_dict` walks `typing.get_type_hints(cls)` and recurses element-wise: nested dataclass fields from sub-tables; `list[T]` / `tuple[T, ...]` / `set[T]` / `dict[K, V]` on their inner annotated types; fixed-length `tuple[T1, T2, ...]` enforces element count (length mismatch raises `ValueError`); `Enum` fields coerced from `.value`; `Union` arms tried in order, first acceptor wins; `Literal[...]` membership enforced (a value outside the declared set raises `ValueError` and names the field); unknown keys raise `TypeError`.
- **Polymorphic discriminator.** A `_neurox_type = "ConcreteConfig"` key inside a sub-table selects a concrete subclass of the annotated base by `__name__`. The key is stripped before the subclass is built. The loader has no table of families — it searches dataclass descendants generically.
- **File round-trip with directives.** `dataclass_from_file(cls, *files, section=None, encoding="utf-8", strict_type=True)` loads each file, expands its directives, plucks `section` (if given), merges in priority order, then coerces. `dataclass_to_file(obj, file, *, encoding="utf-8")` dumps the coerced dict (format chosen by suffix). The supported suffixes are `.toml`, `.yaml`, `.yml`; an unsupported suffix raises `ValueError`.
- **Raw dict access — no coercion, no expansion.** `dict_from_file(file, *, encoding="utf-8")` returns the parsed mapping with directive strings left in place and performs no dataclass coercion; `dict_to_file(data, file, *, encoding="utf-8")` writes a mapping (`None` values are stripped on the TOML path). `encoding` applies to YAML only and is ignored for TOML.
- **Multi-config — one file to many dataclasses.** `dict_configs_from_file(specs, *files, encoding="utf-8", strict_type=True)` maps each `specs` key to a top-level table and builds the paired dataclass from it, returning a dict keyed by section name; a missing section raises `KeyError`. `dict_configs_to_file(objs, file, *, encoding="utf-8")` writes each entry as a top-level table named by its key.
- **Deep overlay merge.** `merge_dicts(*dicts, strict_type=True)` overlays dicts from left (highest priority) to right: sub-tables merge recursively, the higher-priority value wins on a scalar collision, and missing keys are filled from lower priority. With `strict_type=True` a dict-versus-non-dict collision on the same key raises `ValueError`. Inputs are not mutated; a fresh dict is returned.
- **Explicit directive resolution.** `resolve_uses(data, base_dir)` expands every `_neurox_use` / `_neurox_use_preset` in a raw dict and returns a new dict; it is the entry point for callers that loaded via `dict_from_file` and need the expanded form. `_neurox_use` paths resolve against `base_dir`; `_neurox_use_preset` paths resolve against `neurox/presets/`.
- **Directive merge semantics.** A `"<rel_path>:<section>"` reference reads the named section from the resolved file, strips the directive, and merges the fragment under the inline keys via `merge_dicts(inline, fragment)` — inline takes priority. Resolution is recursive (a fragment may itself carry directives, resolved under its own file's directory) and cycle-rejecting (a `(path, section)` re-entry raises `ValueError`). The two directives are mutually exclusive in one sub-table.
- **Resolution order.** Within `dataclass_from_file`, the order is: parse (`dict_from_file`) → directive expansion (`resolve_uses`) → section pluck → multi-file merge (`merge_dicts`) → coercion (`dataclass_from_dict`). Directive expansion therefore precedes both section pluck and the `_neurox_type` dispatch (which runs last, inside coercion).
- **Path resolution base.** `preset_path(rel)` and both preset directives locate the presets root via `importlib.resources.files("neurox") / "presets"`, so editable installs and wheel installs resolve identically. Suffix-free references try `.toml`, then `.yaml`, then `.yml`.
- **Device-fragment purity.** Files under `neurox/presets/process/` carry only physical parameters: one file per device class (`mos.toml`, `rram.toml`, `wire.toml`, ...), one top-level table per variant (`default`, or `<node>_<flavor>`; the class is the filename, not repeated in the section), and no `_neurox_*` keys. The `_neurox_use` / `_neurox_use_preset` / `_neurox_type` keys live in consuming configs, never in the bundled device presets.

## Performance & resources

N/A on the hot path — load and dump are one-time setup. Directive resolution reads each referenced file once per top-level call (a fragment referenced from several sub-tables is not re-read); the per-call read cache is discarded when the call returns.

## Gotchas

- **`dict_from_file` does not expand directives.** It returns the raw mapping with `_neurox_use` / `_neurox_use_preset` strings intact. A caller that needs the expanded tree must call `resolve_uses(data, base_dir)` explicitly; `dataclass_from_file` already does this internally.
- **Inline keys override, not the fragment.** A value set inline alongside a `_neurox_use` wins over the same key in the referenced fragment. Expecting the fragment to be authoritative inverts the contract.
- **Multi-file priority is first-wins.** In `dataclass_from_file(cls, a, b)` and `dict_configs_from_file(specs, a, b)`, file `a` is the higher priority; on a scalar collision `a`'s value survives. Ordering the files wrong silently swaps which value applies.
- **`_neurox_use` inside a preset is an error, not a fallback.** A preset subtree forbids `_neurox_use`; the loader raises rather than resolving it against the user directory. Preset cross-references must use `_neurox_use_preset`.
- **Preset paths reject `..`, leading `./`, and absolute paths at parse time.** A reference outside the forward-relative form fails before any file lookup.
- **`encoding` is YAML-only.** It is accepted but ignored on the TOML path; passing a non-UTF-8 encoding does not affect TOML I/O.

## Known limitations

- **No schema migration or versioning.** The loader fails loud on an unknown key and has no notion of a config-file version or a rename map; adapting an old file to a renamed field is a manual edit, by design (fail-loud over silent coercion).
- **Cycle detection is per top-level resolution.** Cycles are rejected within a single `resolve_uses` / `dataclass_from_file` call via the in-progress `(path, section)` set; there is no persistent cross-call graph, so the guard re-runs from scratch on each load.

---

- **Reference**: N/A — cross-cutting software; the physics it serializes is specified under the per-subsystem reference docs.
- **Implementation**: `neurox/common/load_dump.py`
- **Up-link**: [Config and construction](../config_and_construction.md)
- **Tests**: `tests/test_load_dump_use.py`, `tests/test_config_validation.py`, `tests/test_tool_config.py`
- **Decisions**: [ADR-0001](../../about/adr/ADR-0001-config-dispatch-and-owned-construction.md)
