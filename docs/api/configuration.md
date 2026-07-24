# Configuration and policy

A NeuroX run is configured by two config files passed to the entry points: `--config` (the immutable circuit design) and `--policy` (the nonideality switches). Either file is TOML or YAML, chosen by its suffix.

## The two files

- **`--config`** — the immutable circuit design: tile geometry, device and circuit parameters, and the ADC calibration table. One config file fully specifies a chip and carries no nonideality switches.
- **`--policy`** — the mutable nonideality switches: one `bool` per nonideality source (device mismatch, thermal noise, programming noise, ADC offsets, ...), plus the solver chunking knob. Its section tree mirrors the config.

## Directives

A directive is a reserved key the loader interprets: a field takes a nested mapping carrying a `_neurox_*` key, which a stock TOML or YAML parser reads as ordinary data. The loader recognizes three:

- `_neurox_class = "<Name>Config"` — selects the concrete config class for a polymorphic field; construction dispatches on it. An abstract config base (one that declares the `ABC` signal in its own bases) can never be constructed directly, so every occurrence of it — top-level target, discriminator target, or nested field — must resolve to a concrete class. A concrete class remains constructible even when subclasses of it exist; abstractness is the class's own declaration, never a side effect of which packages are imported.
- `_neurox_use = "<file>:<section>"` — composes in another file or section, so a shared design fragment is written once. The path resolves against the directory of the file that carries the directive.
- `_neurox_use_preset = "<preset>:<section>"` — references a bundled preset under `neurox/presets/` (e.g. `process/rram:default`), resolved against that root wherever the referencing file sits.

The two composition directives are mutually exclusive in one table, and neither may accompany `_neurox_class` — the referenced fragment or preset is the sole owner of its class. Ordinary field values may still be overridden inline beside a directive; inline keys override the fragment, never the reverse.

## Presets

Bundled process presets (device parameters) live under `neurox/presets/`, pulled in via `_neurox_use_preset` and self-describing their own `_neurox_class`. The shared all-off policy preset ships with the scheme, not under `neurox/presets/`; it turns every nonideality off, and a run enables one source by overriding its `bool` inline after the `_neurox_use` line that pulls it in. A config or policy dataclass can also load a bundled preset directly via `Cls.from_preset("family/file:section")`, without a host file.

Resolution enters preset mode at a `_neurox_use_preset` and stays there: inside a preset, `_neurox_use` is rejected, and a preset path may not be absolute, start with `./`, or contain a `..` segment. A preset therefore references only other presets, by a forward path inside the presets root.

## Structure

A file's root must be a table of named sections, and the section tree mirrors the construction tree: a child module's config is a section nested under its owner's, at the field name the owner declares. For the offset 1T1R macro the array owns only the cell and solver; the WL DAC, BL/SL clamp drivers, boundary reference, and readout blocks are the cim_macro's peers above the array. The policy mirrors the same ownership under its own field names.

## Field semantics

A key that matches no field of the target class fails the load. So does a value whose type does not match the field's declared type: a `bool` is not an `int`, and a quoted number is not a number. The one accepted widening is an `int` for a `float` field, so `g_min__uS = 10` and `g_min__uS = 10.0` both load.

Before field construction, the loader normalizes every file to one format-independent value tree: `None`, `bool`, `int`, `float`, and `str` leaves; lists; and mappings with string keys. Values outside that contract are rejected at the file boundary. In particular, quote YAML values that would otherwise be inferred as dates or other YAML-specific Python objects.

Each field's meaning, unit, and Source are documented in the matching subsystem's Reference Parameters section — e.g. the `[cim_macro.array_config]` fields in [reference/primitive/xbar/array/_1t1r/array](../reference/primitive/xbar/array/_1t1r/array.md), with driver and readout fields under the Analog group in [Reference](../reference/README.md). The Source taxonomy (Measured / Process / Design / Calibrated / ...) is defined in [module_parameter](../conventions/module_parameter.md). The runnable end-to-end usage is in the [algorithm-engineer workflow](../guides/algorithm_engineer/workflow.md).

## TOML

Suffix `.toml`. The file is read and written as bytes: TOML is UTF-8 by specification, so the `encoding` argument is inert.

TOML has no null literal, so a `None` value is dropped from a mapping on write and the field round-trips as an absent key. The drop covers mappings only — a `None` inside a list reaches the writer and raises `TypeError`.

## YAML

Suffixes `.yaml` and `.yml`. The file is read and written as text through the `encoding` argument, which defaults to UTF-8; overriding it — or passing `None`, which selects the platform default — changes how a non-ASCII file decodes.

A `None` value is written as `null` and round-trips as `None`.
