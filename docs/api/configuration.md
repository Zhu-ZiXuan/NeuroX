# Configuration and policy

A module tree is built from a paired file set: one file carrying the immutable design, one carrying the run's policy. Each file is TOML or YAML, chosen by its suffix, and each loads into its own dataclass tree independently — nothing in the format binds the two, and the caller forms the pair. Why the split runs where it does, and how the loaded objects then select an implementation, is in [construction](../system_design/construction.md).

## The two files

- **Config** — the immutable circuit design: module geometry, device and circuit parameters, and the calibrated tables the design carries. One config file fully specifies a chip and holds no nonideality switch.
- **Policy** — the run's stance on that design: one `bool` per nonideality source (device mismatch, thermal noise, programming noise, ADC offsets, ...), plus the numerical knobs a run tunes, such as the array's `solve_chunk_size`. Its section tree mirrors the config's.

The entry point names the pair. The bundled examples pass the two files as `--config` and `--policy` (the [algorithm-engineer workflow](../guides/algorithm_engineer/workflow.md) runs them end to end); a calibration tool instead takes one run config that names the pair together with the section to pluck from each, under the rules in [tool conventions](../guides/calibration/tool_conventions.md). The file set a calibration campaign ships, and the provenance tag every physical value in it carries, are specified in [campaigns](../validation/campaigns.md).

## Structure

A load plucks one named section from the file — a dotted name descends nested tables — or takes the file root when it names none. Under that section the tree mirrors the construction tree: a child module's config is a section nested under its owner's, at the field name the owner declares. Ownership, not physical adjacency, fixes that nesting: a peripheral block the macro itself constructs is a section under the macro, beside the array's own, even though it sits electrically at the array's edge. The policy mirrors the same ownership under its own field names.

Several files may feed one object, ordered by descending priority: the merge fills missing keys from the right, so the first file to state a key wins, and a key holding a table in one file and a scalar in another is rejected rather than reconciled.

## Directives

A directive is a reserved key the loader interprets: a field takes a nested mapping carrying a `_neurox_*` key, which a stock TOML or YAML parser reads as ordinary data. The loader recognizes three:

- `_neurox_class = "<Name>Config"` — selects the concrete config class for a polymorphic field; construction dispatches on it, and the name resolves only within that field's own class and its subclasses, so the slot bounds what a file can build. An abstract config base — one declaring the `ABC` signal among its own bases, or still carrying an unimplemented abstract method — can never be constructed, so every slot (top-level target, discriminator target, or nested field) must resolve to a concrete class. A concrete class remains constructible even when subclasses of it exist; abstractness is the class's own declaration, never a side effect of which packages are imported.
- `_neurox_use = "<file>:<section>"` — composes in another file's section, so a shared design fragment is written once. The path resolves against the directory of the file that carries the directive, and a path written without a suffix tries `.toml`, then `.yaml`, then `.yml`.
- `_neurox_use_preset = "<preset>:<section>"` — the same composition against a bundled preset under `neurox/presets/` (e.g. `process/rram:default`), resolved against that root wherever the referencing file sits.

Both composition directives name a non-empty path and a non-empty section, and a chain that arrives back at a section it is already resolving is rejected as a cycle rather than followed.

The two composition directives are mutually exclusive in one table, and neither may accompany `_neurox_class` — the referenced fragment or preset is the sole owner of its class. Ordinary field values may still be overridden inline beside a directive; inline keys override the fragment, never the reverse.

## Presets

`neurox/presets/` holds the bundled process presets — the device parameter sets a design draws on — pulled in via `_neurox_use_preset` and self-describing their own `_neurox_class`. A config or policy dataclass can also load one directly via `Cls.from_preset("family/file:section")`, without a host file.

Resolution enters preset mode at a `_neurox_use_preset` and stays there: inside a preset, `_neurox_use` is rejected, and a preset path may not be absolute, start with `./`, or contain a `..` segment. A preset therefore references only other presets, by a forward path inside the presets root.

## Field semantics

A config or policy field carries no default: the file states every field of the class it builds, and a load that omits some fails, naming which. A key matching no field of the target class fails the same way, naming the fields that class does declare.

A value whose type does not match the field's declared type fails too: a `bool` is not an `int`, and a quoted number is not a number. The one accepted widening is an `int` for a `float` field, so `g_min__uS = 10` and `g_min__uS = 10.0` both load. Beyond the primitives, a field declared over a named set of choices takes one choice by its value (`w_encoding = "true_form"`), a fixed-length tuple field takes a list of exactly that length (`x_value_range = [0, 15]`), and a field declared as a filesystem path takes a string.

Before field construction, the loader normalizes every file to one format-independent value tree: `None`, `bool`, `int`, `float`, and `str` leaves; lists; and mappings with string keys. Values outside that contract are rejected at the file boundary. In particular, quote YAML values that would otherwise be inferred as dates or other YAML-specific Python objects.

Each field's meaning, unit, and Source are documented in the matching subsystem's Reference Parameters section — e.g. the `[cim_macro.array_config]` fields in [reference/primitive/xbar/array/1t1r](../reference/primitive/xbar/array/1t1r.md), with driver and readout fields under the Analog group in [Reference](../reference/README.md). The Source taxonomy (Measured / Process / Design / Calibrated / ...) is defined in [module_parameter](../conventions/module_parameter.md).

## TOML

Suffix `.toml`. The file is read and written as bytes: TOML is UTF-8 by specification, so the `encoding` argument is inert.

TOML has no null literal, so a `None` value is dropped from a mapping on write and the field round-trips as an absent key. The drop covers mappings only — a `None` inside a list reaches the writer and fails.

## YAML

Suffixes `.yaml` and `.yml`. The file is read and written as text through the `encoding` argument, which defaults to UTF-8; overriding it changes how a non-ASCII file decodes.

A `None` value is written as `null` and round-trips as `None`.
