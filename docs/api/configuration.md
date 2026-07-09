# Configuration and policy

A NeuroX run is configured by two TOML files passed to the entry points: `--config` (the immutable circuit design) and `--policy` (the nonideality switches). They are separate on purpose — the config fully specifies a chip's physical design; the policy selects which nonidealities are active for a run.

## The two files

- **`--config`** — the immutable circuit design: tile geometry, device and circuit parameters, and the ADC calibration table. One config file fully specifies a chip and carries no nonideality switches.
- **`--policy`** — the mutable nonideality switches: one `bool` per nonideality source (device mismatch, thermal noise, programming noise, ADC offsets, ...), plus the solver chunking knob. Its section tree mirrors the config.

## Directives

The loader recognizes three directives:

- `_neurox_type = "<Name>Config"` — selects the concrete config class for a polymorphic field; construction dispatches on it.
- `_neurox_use = "<file>[:<section>]"` — composes in another TOML file or section, so a shared design fragment is written once.
- `_neurox_use_preset = "<preset>[:<section>]"` — references a bundled preset under `neurox/presets/` (e.g. `process/rram:default`).

## Presets

Bundled process presets (device parameters) live under `neurox/presets/`, pulled in via `_neurox_use_preset`. The shared all-off policy preset ships with the scheme, not under `neurox/presets/`; it turns every nonideality off, and a run enables one source by overriding its `bool` inline after the `_neurox_use` line that pulls it in.

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
