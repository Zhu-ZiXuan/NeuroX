# Configuration and policy

Configuration and policy load independently from TOML or YAML files into dataclass trees. The caller pairs the hardware design with the run choices under [construction](../system_design/construction.md).

Convolution geometry (`stride`, `padding`, `dilation`, and `groups`) accompanies `w_logical_shape` as constructor context, including in `conv2d_unit_from_file`.

## The two files

- **Config** — the immutable circuit design: module geometry, device and circuit parameters, and calibrated tables. It fully specifies a chip and holds no nonideality switches.
- **Policy** — one `bool` per nonideality source, plus numerical settings such as `solve_chunk_size`. Its section tree mirrors the config's.

Each entry point selects the pair. Calibration uses one run config naming both files and their sections, under [tool conventions](../guides/calibration/tool_conventions.md); [campaigns](../validation/campaigns.md) defines the campaign file set and physical-value provenance tags.

## Structure

A load selects a dot-separated section or, when omitted, the file root. Config and policy trees follow ownership: each child is nested under its owner's declared field name, independently of physical adjacency.

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

## Field semantics

Every config or policy field is required. Loading rejects missing fields and unknown keys, naming the missing or accepted fields.

Values must match declared types: a `bool` cannot fill an `int` field, and quoted numbers are strings. The only numeric widening is `int` to `float`, so `g_min__uS = 10` and `g_min__uS = 10.0` both load. Enum fields take a choice's value (`w_encoding = "true_form"`), fixed-length tuples take lists of that length (`x_value_range = [0, 15]`), and filesystem paths take strings.

Files normalize to `None`, `bool`, `int`, `float`, and `str` leaves, lists, and mappings with string keys. Other values are rejected before field construction; quote YAML dates and other values that would become format-specific objects.

The owning field docstring defines its interface semantics. Model parameters, units, constraints, and provenance belong to the corresponding Reference document under the [parameter Source taxonomy](../conventions/module_parameter.md).

## TOML

Files use `.toml` and UTF-8; the `encoding` argument has no effect.

TOML has no null literal. Writing drops `None` mapping entries; `None` inside a list fails.

## YAML

Files use `.yaml` or `.yml` and the requested `encoding`, which defaults to UTF-8.

A `None` value is written as `null` and round-trips as `None`.
