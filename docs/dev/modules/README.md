# Module Docs

This directory mirrors the source tree and records the current design of specific NeuroX packages and files.

The on-disk layout below `modules/` is intended to be **one-to-one** with `neurox/`. Each Python module gets its own markdown file; each package gets a `README.md`.

Current documented areas:

- `common/` — config base + dispatch mixin + load/dump + noise + constants
- `device/` — NMOS, RRAM, Selector
- `analog/` — clamp-driver protocol, AnalogMux, Decoder, Driver, SwitchCap, and the `dac/`, `adc/`, `tia/`, `readout/` polymorphic families
- `digital/` — accumulator, adders, shift-adder, subtractor
- `xbar/` — abstract `Xbar`, ideal twin, shared solver primitives, and the `_1t1r/` subtree (`circuit_core`, `newton_raphson_solver`, `offset`, `differential` / `simple_core` placeholders)
- `mapper/` — `transcoder` plus the `xbar/slicer/` value-domain primitive
- `macro/` — `XbarMacro` (abstract) + concrete modes (`DirectXbarMacro`, `InterArraySliceXbarMacro`, `IntraArraySliceXbarMacro`) + `IdealMacro`
- `operator/` — quantised operator surface + `train/` HAT path
- `profiler/` — `ProfileMixin` + `NeuroxProfiler` side channel
- `replace/` — model-rewriting pipeline
- `tools/` — offline calibration / analysis CLIs

When a module family is refactored, its long-lived design notes should move here rather than staying in code docstrings.
