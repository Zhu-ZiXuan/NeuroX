# Module Docs

This directory mirrors the source tree and records the current design of specific NeuroX packages and files.

The on-disk layout below `modules/` is intended to be **one-to-one** with `neurox/`. Each Python module gets its own markdown file; each package gets a `README.md`.

Current documented areas:

- `common/` — `CircuitBase` + `CircuitConfig` typed PPA root, config base + dispatch mixin + load/dump + noise + constants
- `device/` — NMOS, RRAM, Selector
- `analog/` — clamp-driver protocol, AnalogMux, Driver, SwitchCap, and the `dac/`, `adc/`, `tia/` polymorphic families
- `digital/` — accumulator, adders, shift-adder, subtractor
- `xbar/` — abstract `Xbar`, ideal twin, shared solver primitives, the `_1t1r/` subtree (`circuit_core`, `solver`, `nested_solver`, `full_jacobian_solver`, `offset`), and the `readout/` polymorphic family
- `mapper/` — `transcoder` plus the `xbar/slicer/` value-domain primitive
- `macro/` — `XbarMacro` (abstract registry root) + every concrete subclass
- `profiler/` — `ProfileMixin` + `NeuroxProfiler` side channel
- `tools/` — offline calibration / analysis CLIs (`calculate_1t1r_states`, `xbar_adc/{statistic,calibrate}`, `xbar_tia/optimize`, `solver_calibrate/{tia,nested,full_jacobian}`)

When a module family is refactored, its long-lived design notes should move here rather than staying in code docstrings.
