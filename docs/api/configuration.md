# Configuration and policy

A NeuroX run is configured by two TOML files passed to the entry points: `--config` (the immutable circuit design) and `--policy` (the nonideality switches). They are separate on purpose — the config fully specifies a chip's physical design; the policy selects which non-idealities are active for a run.

## The two files

- **`--config`** — the immutable circuit design: tile geometry, device and circuit parameters, and the ADC calibration table. One config file fully specifies a chip and carries no nonideality switches.
- **`--policy`** — the mutable nonideality switches: one `bool` per non-ideality source (device mismatch, thermal noise, programming noise, ADC offsets, ...), plus the solver chunking knob. Its section tree mirrors the config.

## Directives

The loader recognizes three directives:

- `_neurox_type = "<Name>Config"` — selects the concrete config class for a polymorphic field; construction dispatches on it.
- `_neurox_use = "<file>[:<section>]"` — composes in another TOML file or section, so a shared design fragment is written once.
- `_neurox_use_preset = "<preset>[:<section>]"` — references a bundled preset under `neurox/presets/` (e.g. `policy/all_off:macro`).

## Presets

Bundled presets live under `neurox/presets/`. The shared all-off policy preset (`policy/all_off.toml`) turns every non-ideality off; a run enables one source by overriding its `bool` inline after the `_neurox_use_preset` line.

## Structure

The TOML section tree mirrors the construction tree. For the offset 1T1R macro the config nests as:

```toml
[xbar]
[xbar.core_config]
[xbar.core_config.cell_config]
[xbar.core_config.solver_config]
[xbar.wl_dac_config]
[xbar.tia_config]
[xbar.sl_driver_config]
[xbar.clamp_ref_config]
[xbar.signal_switchcap_config]
[xbar.ref_switchcap_config]
[xbar.voltage_mux_config]
[xbar.adc_config]
[xbar.adc_v_ref_config]
```

The core array owns only the cell and solver; the WL DAC, BL/SL clamp drivers, boundary reference, and readout blocks are the xbar's peers above the core. The policy mirrors the same ownership:

```toml
[policy]
[policy.xbar.core.cell]
[policy.xbar.tia]
```

## Field semantics

Each field's meaning, unit, and Source are documented in the matching subsystem's Reference Parameters section — e.g. the `[xbar.core_config]` fields in [reference/xbar/_1t1r/core](../reference/xbar/_1t1r/core.md), and the driver / inline-readout fields (WL DAC, BL/SL clamp drivers, boundary reference, signal / ref switch-cap, voltage mux, ADC, ADC-ladder reference) under your scheme xbar and the [analog leaves](../reference/analog/README.md). Provenance terms (Measured / Process / Design / Calibrated / ...) are defined in [module_parameter](../conventions/module_parameter.md). The runnable end-to-end usage is in the [algorithm-engineer guide](../guides/algorithm_engineer/README.md).
