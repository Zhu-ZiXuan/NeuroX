# Load / dump

## Summary

`load_dump` is the format boundary between NeuroX's frozen `*Config` dataclass tree and plain TOML / YAML files: it (de)serializes dataclasses to and from config files, and layers on that plain round-trip NeuroX's own config-file directive mechanism — a cross-file include (`_neurox_use`) and a bundled-preset reference (`_neurox_use_preset`). The (de)serialization and merge helpers are ordinary tools whose contract is stated in their own docstrings; this record keeps only the code-invisible *why* of the directive and preset mechanism. It is cross-cutting software utility, carrying no physics and owning no reference twin.

## Design decisions

- **Directives live in ordinary string keys, not a custom file dialect.** A directive is an ordinary string-valued key (`_neurox_use = "<path>:<section>"`); a stock TOML / YAML parser reads it as plain data and never has to understand NeuroX. All include semantics live in this module's resolution pass, so a config file stays loadable, diffable, and editable with off-the-shelf tooling, and the mechanism adds no grammar to the file format.
- **Two directives, split by which base directory resolves the path.** `_neurox_use` resolves its path relative to the file that carries the directive; `_neurox_use_preset` resolves relative to the bundled presets root. The split tracks ownership: a user-authored fragment travels and versions with the referencing project and must be found beside it, whereas a bundled reference parameter travels and versions with the package and must be found in the package no matter where the referencing file lives. A single shared base cannot honour both — resolving a preset against the referencing file would let a config that is moved or copied silently re-bind the preset to whatever happens to sit in the new directory.
- **A preset reference opens a closed, forward-only dependency graph.** Once resolution enters a fragment through `_neurox_use_preset` it stays in preset mode, where a nested `_neurox_use` is rejected, and a preset path may not be absolute, start with `./`, or contain a `..` segment. Together these confine every transitive reference of a preset to a forward path inside the presets root: a preset can depend only on other presets, never on a user file, and no reference can climb out of the package tree. A bundled preset's full dependency closure is therefore reproducible from the package alone, and relocating the referencing user file cannot change what a preset resolves to.
- **Inline keys override the fragment, never the reverse.** A directive contributes a base table; the keys written inline beside the directive are layered on top and win every collision. A reference thus reads as "start from this fragment, then patch these fields" — the usual authoring idiom — and one shared fragment can back many call sites that each adjust a few values without editing the fragment.
- **An unknown key fails loud instead of silently defaulting.** A key that matches no field of the target dataclass raises rather than being dropped. Silently ignoring it would let a misspelled parameter fall back to its default and ship a wrong number behind a config that looks correct; failing at load time surfaces the typo where it can be fixed. The same stance rules out a silent schema-migration layer — there is no config version tag or field-rename map, so adapting a file written against an old field name is a deliberate manual edit, not an automatic and invisible coercion.

---

- **Reference**: N/A — software utility
- **Implementation**: `neurox/common/load_dump.py`
- **Tests**: `tests/test_load_dump_use.py`
