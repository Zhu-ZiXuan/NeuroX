# Configuration and Policy

A NeuroX run is configured by two TOML files passed to the entry points: `--config` (the immutable circuit design) and `--policy` (the nonideality switches). They are separate on purpose — the config fully specifies a chip's physical design; the policy selects which non-idealities are active for a run.

## The two files

- **`--config`** — the immutable circuit design: tile geometry, device and circuit parameters, and the ADC calibration table. One config file fully specifies a chip and carries no nonideality switches.
- **`--policy`** — the mutable nonideality switches: one `bool` per non-ideality source (device mismatch, thermal noise, programming noise, ADC offsets, ...), plus the solver chunking knobs. Its section tree mirrors the config.

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
[xbar.core_config.solver_config]
[xbar.readout_config]
```

and the policy mirrors the same ownership:

```toml
[policy]
[policy.xbar.core.tia]
```

## Field semantics

Each field's meaning, unit, and Source are documented in the matching subsystem's Reference Parameters section — e.g. the `[xbar.core_config]` fields in [reference/xbar/_1t1r/circuit_core](../reference/xbar/_1t1r/circuit_core.md), the readout fields in [reference/xbar/readout](../reference/xbar/readout/README.md). Provenance terms (Measured / Process / Design / Calibrated / ...) are defined in [parameter_provenance](../reference/parameter_provenance.md). The runnable end-to-end usage is in the [algorithm-engineer guide](../guides/algorithm_engineer/README.md).
