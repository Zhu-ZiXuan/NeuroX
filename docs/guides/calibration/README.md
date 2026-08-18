# Calibration

Offline runs that seat a chip config's numerical values against hardware. Each one reads a run config, drives a configured macro outside the forward path, and emits a config fragment to paste back.

- [Tool conventions](tool_conventions.md) — the CLI, run-config, output, and ordering rules every calibration command shares. Start here.
- [Solver iteration counts](solver_iteration_counts.md) — the per-cell condensation count, the linearized-cell extraction, and the array solver's iteration pair.
- [ADC calibration](calibrate_adc.md) — the quantization mode set, the analog threshold ladder, and the per-mode rescale factor.
