# `neurox/common/config.py`

## Current role

`config.py` carries the two oldest config primitives:

- `Config` — the optional base class that gives a frozen dataclass `from_file(...)` / `to_file(...)` plus a `from_builder(...)` bridge.
- `ConfigBuilder` — a generic staged-construction base for configs that need validation across multiple optional fields before they can be finalised.

The polymorphic-family dispatch mixin lives in [`config_dispatch.py`](config_dispatch.md), not here.

## File-IO layer

`Config.from_file(...)` is a convenience wrapper around [`load_dump.dataclass_from_file`](load_dump.md). It accepts one or more TOML / YAML files in descending-priority order and an optional `section=` argument for multi-config files. The base class itself does not implement the parsing — it forwards to `load_dump`.

## When to subclass

Most NeuroX config dataclasses do **not** inherit `Config` today. They are plain `@dataclass(frozen=True)` types and rely on `load_dump.dataclass_from_file(SomeConfig, path, section=...)` instead. The `Config` base only adds method-style I/O ergonomics and is retained for forwards-compatibility with builder-based flows.

## `ConfigBuilder`

`ConfigBuilder[T]` is a generic helper for configs whose validation is non-trivial:

- subclasses are dataclasses with `Optional[...]` fields
- they implement `validate(self) -> None` (raising `ValidationError`)
- they implement `build(self) -> T` to produce the final immutable config

Builders are useful when individual fields are filled in at different points in time and validation must run only once everything is known. NeuroX's hot config paths do not currently use builders, but the surface is kept so future configuration UIs can layer on top.

See also:

- `../../architecture/config_and_construction.md`
- `config_dispatch.md`
- `load_dump.md`
