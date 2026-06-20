# NeuroX

NeuroX is a static-PPA + functional-accuracy co-simulation framework for memristor (1T1R RRAM) crossbar-based AI accelerators. It models the analog VMM signal chain — RRAM cell, access NMOS, BL/SL/WL wire parasitics, TIA clamp, sample-and-hold, mux, and ADC — and lifts the integer output through digital aggregation (shift-add, accumulators) up to a macro layer that exposes one `matmul` per fabricated tile.

## Public surface

The core library's public surface stops at `neurox.macro`. Above that line — training loops, observer calibration, model rewriting — lives in `example/` and is not a stable API. Below it, every primitive (`Xbar`, `Solver`, `OpAmpTIA`, `McsSarAdc`, …) follows the same family-base + concrete-config + `from_config` registry pattern documented in [`docs/internals/config_and_construction.md`](docs/internals/config_and_construction.md).

## Layout

- `neurox/device/` — RRAM, NMOS, Selector device-physics primitives.
- `neurox/analog/` — clamp driver, AnalogMux, SwitchCap, Driver, `dac/` / `adc/` / `tia/` polymorphic families.
- `neurox/digital/` — integer accumulators, shift-adders, subtractors.
- `neurox/xbar/` — abstract `Xbar`, `IdealXbar` reference twin, the `_1t1r/` subtree (circuit core + DC solver + offset-coded readout), and the `readout/` family.
- `neurox/mapper/` — value-domain transcoder and slicer primitives.
- `neurox/macro/` — `XbarMacro` family with `from_config` factory; the public entry point.
- `neurox/profiler/` — `ProfileMixin` and the side-channel dynamic-energy logger.
- `neurox/tools/` — offline calibration / analysis CLIs (state map, ADC stat / calibrate, TIA optimize, solver calibrate).
- `example/` — runnable LeNet and BERT pipelines (float train, HAT, evaluation) showing how a user assembles the above into a training/inference flow.

## Example pipelines

See [`docs/guides/algorithm_engineer/`](docs/guides/algorithm_engineer/README.md) for the LeNet and BERT walkthroughs.

## Developer docs

See [`docs/`](docs/README.md) — the scientific reference, implementation internals, contributing standards, and ADRs.
