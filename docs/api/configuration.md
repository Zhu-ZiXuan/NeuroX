# Configuration and policy

Configuration and policy load independently from TOML or YAML files into dataclass trees. The caller pairs the hardware design with the run choices under [construction](../system_design/construction.md).

Convolution geometry (`stride`, `padding`, `dilation`, and `groups`) is constructor context supplied with `w_logical_shape`, rather than fields in the hardware config. `conv2d_unit_from_file` also receives this geometry explicitly from its caller.

## The two files

- **Config** — the immutable circuit design: module geometry, device and circuit parameters, and the calibrated tables the design carries. One config file fully specifies a chip and holds no nonideality switch.
- **Policy** — the run's stance on that design: one `bool` per nonideality source (device mismatch, thermal noise, programming noise, ADC offsets, ...), plus the numerical knobs a run tunes, such as the array's `solve_chunk_size`. Its section tree mirrors the config's.

Each entry point selects the pair. Calibration uses one run config naming both files and their sections, under [tool conventions](../guides/calibration/tool_conventions.md); [campaigns](../validation/campaigns.md) defines the campaign file set and physical-value provenance tags.

## Structure

A load selects a named section, using dots to descend nested tables, or the file root when no section is given. Config and policy trees mirror ownership: each child is nested under its owner at the declared field name, independently of physical adjacency.

Several files may feed one object, ordered by descending priority: the merge fills missing keys from the right, so the first file to state a key wins, and a key holding a table in one file and a scalar in another is rejected rather than reconciled.

## Directives

The loader interprets reserved `_neurox_*` keys in nested mappings:

- `_neurox_class = "<Name>Config"` — selects the config class for a polymorphic field. The name resolves only within the declared class and its descendants; without the directive, the loader uses the declared class. This applies to both top-level targets and nested fields. The selected class is constructed normally, including its own construction constraints and validation.
- `_neurox_use = "<file>:<section>"` — composes in another file's section, so a shared design fragment is written once. The path resolves against the directory of the file that carries the directive, and a path written without a suffix tries `.toml`, then `.yaml`, then `.yml`.
- `_neurox_use_preset = "<preset>:<section>"` — the same composition against a bundled preset under `neurox/presets/` (e.g. `process/rram:default`), resolved against that root wherever the referencing file sits.

Both composition directives name a non-empty path and a non-empty section, and a chain that arrives back at a section it is already resolving is rejected as a cycle rather than followed.

The two composition directives are mutually exclusive in one table, and neither may accompany `_neurox_class` — the referenced fragment or preset is the sole owner of its class. Ordinary field values may still be overridden inline beside a directive; inline keys override the fragment, never the reverse.

## Presets

`neurox/presets/` holds bundled parameter sets: process fragments under `process/` and published design points under `works/`. They can be pulled in via `_neurox_use_preset` and self-describe their own `_neurox_class`. A config or policy dataclass can also load one directly via `Cls.from_preset("family/file:section")`, without a host file.

Resolution enters preset mode at a `_neurox_use_preset` and stays there: inside a preset, `_neurox_use` is rejected, and a preset path may not be absolute, start with `./`, or contain a `..` segment. A preset therefore references only other presets, by a forward path inside the presets root.

## Field semantics

A config or policy field carries no default: the file states every field of the class it builds, and a load that omits some fails, naming which. A key matching no field of the target class fails the same way, naming the fields that class does declare.

A value whose type does not match the field's declared type fails too: a `bool` is not an `int`, and a quoted number is not a number. The one accepted widening is an `int` for a `float` field, so `g_min__uS = 10` and `g_min__uS = 10.0` both load. Beyond the primitives, a field declared over a named set of choices takes one choice by its value (`w_encoding = "true_form"`), a fixed-length tuple field takes a list of exactly that length (`x_value_range = [0, 15]`), and a field declared as a filesystem path takes a string.

Before field construction, the loader normalizes every file to one format-independent value tree: `None`, `bool`, `int`, `float`, and `str` leaves; lists; and mappings with string keys. Values outside that contract are rejected at the file boundary. In particular, quote YAML values that would otherwise be inferred as dates or other YAML-specific Python objects.

The owning field docstring defines its interface semantics. Model parameters, units, constraints, and provenance belong to the corresponding Reference document under the [parameter Source taxonomy](../conventions/module_parameter.md).

## TOML

Suffix `.toml`. The file is read and written as bytes: TOML is UTF-8 by specification, so the `encoding` argument is inert.

TOML has no null literal, so a `None` value is dropped from a mapping on write and the field round-trips as an absent key. The drop covers mappings only — a `None` inside a list reaches the writer and fails.

## YAML

Suffixes `.yaml` and `.yml`. The file is read and written as text through the `encoding` argument, which defaults to UTF-8; overriding it changes how a non-ASCII file decodes.

A `None` value is written as `null` and round-trips as `None`.
