# `neurox/common/load_dump.py`

## Current role

`load_dump.py` converts between NeuroX's frozen dataclass configs and plain TOML / YAML files. It is the only module that knows about file formats; every other config-loading entry point routes through it.

## Public surface

- `dataclass_from_dict(cls, data)` / `dataclass_to_dict(obj)` — in-memory round-trip between a dataclass tree and nested dicts.
- `dataclass_from_file(cls, *files, section=..., encoding=..., strict_type=...)` / `dataclass_to_file(obj, file, encoding=...)` — TOML / YAML file round-trip; resolves `_neurox_use` directives.
- `dict_from_file` / `dict_to_file` — raw dict access without dataclass coercion or `_neurox_use` expansion.
- `dict_configs_from_file` / `dict_configs_to_file` — multi-config file helpers where each top-level table maps to a distinct dataclass.
- `merge_dicts` — deep overlay merging multiple dicts in priority order.
- `resolve_uses(data, base_dir)` — explicit `_neurox_use` resolver for raw-dict callers.

## NeuroX-private extension keys

The loader recognises two NeuroX-private string-valued keys:

- `_neurox_type` — polymorphic-dataclass discriminator
- `_neurox_use` — cross-file fragment reference

External TOML / YAML parsers treat them as ordinary strings.

## Recursive dataclass coercion

`dataclass_from_dict(cls, data)` walks `cls`'s `typing.get_type_hints` and recurses element-wise:

- nested dataclass fields are built from nested dict sub-tables
- `list[T]` / `tuple[T, ...]` / `set[T]` / `dict[K, V]` fields recurse on their inner annotated types
- `Enum` fields are coerced from their `.value`
- `Union` arms are tried in order; the first one that accepts the value wins
- unknown keys are silently dropped so older config files stay compatible

## Polymorphic `_neurox_type` discriminator

When a dataclass field is annotated with an abstract / base class, the loader recognises a `_neurox_type = "ConcreteSubclassConfig"` key inside the sub-table and instantiates the named subclass instead of the annotated type. The discriminator key is stripped from the sub-table before the subclass is built; the loader walks the subclass tree to locate a dataclass with the matching `__name__`.

The loader has no knowledge of which polymorphic families exist — it operates on dataclass subclasses generically.

## `Literal[...]` enforcement

`Literal` annotations are enforced at deserialisation time: a value not present in the declared set raises `ValueError` and identifies the offending field.

## Multi-file merge

`dataclass_from_file(cls, *files, section=...)` deep-merges the supplied files in priority order (first wins on key collisions) and then builds `cls` from the merged dict. Sub-tables are merged recursively; non-dict values are replaced by the higher-priority entry.

## `_neurox_use` cross-file references

Any sub-table may carry a `_neurox_use = "<rel_path>:<section>"` directive. The loader reads the named section from `<rel_path>` (resolved relative to the file containing the directive; `.toml`/`.yaml`/`.yml` are tried when no suffix is given), strips `_neurox_use` from the sub-table, and merges the fragment with the remaining inline keys. **Inline keys override the fragment** — the directive supplies defaults, the call site customises.

`_neurox_use` resolves recursively (a fragment may itself contain `_neurox_use`) and rejects cycles. Resolution happens before the `_neurox_type` discriminator dispatch and before section pluck.

`resolve_uses(data, base_dir)` is exposed for callers that load via `dict_from_file` and need the expanded form. `dict_from_file` itself returns the raw dict, leaving `_neurox_use` strings in place.

### Device-fragment purity rule

Files under `neurox/config/process/` carry only physical parameters:

- one file per device class (`mos.toml`, `rram.toml`, `wire.toml`, …)
- one top-level table per variant, named `<class>_<node>_<flavor>`
- no `_neurox_*` keys

`_neurox_use` / `_neurox_type` live in the consuming macro / architecture configs.

See also:

- `config.md`
- `config_dispatch.md`
- `../../architecture/config_and_construction.md`
