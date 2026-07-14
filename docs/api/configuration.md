# Configuration and policy

A NeuroX run is configured by two TOML files passed to the entry points: `--config` (the immutable circuit design) and `--policy` (the nonideality switches).

## The two files

- **`--config`** — the immutable circuit design: tile geometry, device and circuit parameters, and the ADC calibration table. One config file fully specifies a chip and carries no nonideality switches.
- **`--policy`** — the mutable nonideality switches: one `bool` per nonideality source (device mismatch, thermal noise, programming noise, ADC offsets, ...), plus the solver chunking knob. Its section tree mirrors the config.

## Directives

The loader recognizes three directives:

- `_neurox_class = "<Name>Config"` — selects the concrete config class for a polymorphic field; construction dispatches on it. A polymorphic base (one with dataclass subclasses) can never be constructed directly, so every occurrence of it — top-level target, discriminator target, or nested field — must resolve to a concrete leaf.
- `_neurox_use = "<file>:<section>"` — composes in another TOML file or section, so a shared design fragment is written once.
- `_neurox_use_preset = "<preset>:<section>"` — references a bundled preset under `neurox/presets/` (e.g. `process/rram:default`).

A table carrying `_neurox_use` or `_neurox_use_preset` must not also declare `_neurox_class` — the referenced fragment or preset is the sole owner of its class; ordinary field values may still be overridden inline beside the directive. A bundled preset pulls straight into a nested field either as an inline table, `field = { _neurox_use_preset = "process/rram:default" }`, or as the equivalent TOML dotted-key form, `field._neurox_use_preset = "process/rram:default"` — both parse to the same nested mapping.

## Presets

Bundled process presets (device parameters) live under `neurox/presets/`, pulled in via `_neurox_use_preset` and self-describing their own `_neurox_class`. The shared all-off policy preset ships with the scheme, not under `neurox/presets/`; it turns every nonideality off, and a run enables one source by overriding its `bool` inline after the `_neurox_use` line that pulls it in. A config or policy dataclass can also load a bundled preset directly via `Cls.from_preset("family/file:section")`, without a host file.

## Structure

The TOML section tree mirrors the construction tree. For the offset 1T1R macro the config nests as:

```toml
[cim_macro]
[cim_macro.array_config]
[cim_macro.array_config.cell_config]
[cim_macro.array_config.solver_config]
[cim_macro.wl_dac_config]
[cim_macro.tia_config]
[cim_macro.sl_driver_config]
[cim_macro.clamp_ref_config]
[cim_macro.signal_switchcap_config]
[cim_macro.ref_switchcap_config]
[cim_macro.voltage_mux_config]
[cim_macro.adc_config]
[cim_macro.adc_v_ref_config]
```

The array owns only the cell and solver; the WL DAC, BL/SL clamp drivers, boundary reference, and readout blocks are the cim_macro's peers above the array. The policy mirrors the same ownership:

```toml
[policy]
[policy.cim_macro.array.cell]
[policy.cim_macro.tia]
```

## Field semantics

Each field's meaning, unit, and Source are documented in the matching subsystem's Reference Parameters section — e.g. the `[cim_macro.array_config]` fields in [reference/primitive/xbar/array/_1t1r/array](../reference/primitive/xbar/array/_1t1r/array.md), and the driver / inline-readout fields (WL DAC, BL/SL clamp drivers, boundary reference, signal / ref switch-cap, voltage mux, ADC, ADC-ladder reference) under your scheme cim_macro and the [analog leaves](../reference/primitive/analog/README.md). The Source taxonomy (Measured / Process / Design / Calibrated / ...) is defined in [module_parameter](../conventions/module_parameter.md). The runnable end-to-end usage is in the [algorithm-engineer guide](../guides/algorithm_engineer/README.md).
