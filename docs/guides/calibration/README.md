# Calibration

Offline runs that seat a chip config's numerical values against hardware. Each command reads a reproducible run config and either reports the evidence for a manual choice or emits a config fragment to paste back.

- [Tool conventions](tool_conventions.md) — the CLI, run-config, output, and ordering rules every calibration command shares. Start here.
- [Cell linearization and solver contract](solver_tolerances.md) — extract a linearized cell where needed and understand the ordinary and diagnostic solve contracts.
- [ADC input characterization](calibrate_adc.md) — report nominal per-ideal-value input clusters for manual reference selection.
- [Macro calibration](calibrate_macro.md) — derive quantization modes and determine each mode's full-resolution output rescale factor.
