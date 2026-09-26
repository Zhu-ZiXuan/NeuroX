# Configuration and policy

Configuration and policy load independently from TOML or YAML files into dataclass trees. The caller pairs the hardware design with the run choices under [construction](../system_design/construction.md).

Convolution geometry (`stride`, `padding`, `dilation`, and `groups`) accompanies `w_logical_shape` as constructor context, including in `conv2d_unit_from_file`.

## The two files

- **Config** — the immutable circuit design: module geometry, device and circuit parameters, and calibrated tables. It specifies a component and its owned children.
- **Policy** — one `bool` per nonideality source, plus numerical settings such as `solve_chunk_size`. Its section tree mirrors the config's.

Calibration run files name both bindings and their sections; see [running calibration tools](../guides/calibration/tool_conventions.md) for invocation and artifacts.

## Structure

A load selects a section from each file before merging; omitting it selects each file root. Config and policy trees follow ownership: each child is nested under its owner's declared field name, independently of physical adjacency.

When several files feed one object, the first file to state a key wins. A table/scalar conflict at the same key is rejected.

## Directives

The loader interprets reserved `_neurox_*` keys in nested mappings:

- `_neurox_class = "<Name>Config"` — selects the declared class or one of its descendants for a top-level target or nested polymorphic field. Without it, the declared class is used. The selected class's construction constraints and validation apply.
- `_neurox_use = "<file>:<section>"` — composes another file's section. Paths resolve relative to the referencing file; suffixless paths try `.toml`, `.yaml`, then `.yml`.
- `_neurox_use_preset = "<preset>:<section>"` — composes a bundled preset relative to `neurox/presets/`, such as `process/rram:default`.

Both composition directives require a non-empty path and section. Cyclic references are rejected.

The composition directives are mutually exclusive and cannot accompany `_neurox_class`; the referenced fragment owns its class. Inline field values override the fragment.

## Presets

`neurox/presets/` holds process fragments under `process/` and published design points under `works/`, each declaring its `_neurox_class`. Load them with `_neurox_use_preset` or directly with `Cls.from_preset("family/file:section")`.

Presets can reference only other presets. Within them, `_neurox_use` is rejected, and paths cannot be absolute, start with `./`, or contain a `..` segment.

## Typed loading and file output

Loaded fields are checked against the selected configuration or policy schema. Use the [serialization API](extensions.md#neurox.common.serialize_mixin.SerializeMixin) for field conversion, validation errors, per-file section selection, and writing behavior. Individual field docstrings define interface constraints; the corresponding scientific reference explains model parameters and their provenance.

TOML uses UTF-8 and has no null literal. YAML can represent null values. The loader and writer functions in the [serialization tools](extensions.md#neurox.common.serialize) document supported values and format-specific handling.
