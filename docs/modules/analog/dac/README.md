# DAC Family

The `DAC` family converts integer codes to analog voltages. It follows the project-wide polymorphic-family pattern (see [`docs/dev/architecture/config_and_construction.md`](../../../dev/architecture/config_and_construction.md)).

Rules:

- `DACConfig` is the family base config.
- Each concrete DAC owns a concrete `DACConfig` subclass.
- `DAC.from_config(...)` dispatches by `type(config)` via the family registry.
- DAC has no family-specific runtime extras today; its `from_config` signature is the canonical `(config, name, T__K, dtype)`.

Current concrete implementation:

- `GeneralDAC` — LUT-based DAC with optional Gaussian drive-thermal noise.
