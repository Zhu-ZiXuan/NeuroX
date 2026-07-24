# NeuroX

NeuroX is a static-PPA + functional-accuracy co-simulation framework for memristor (1T1R RRAM) crossbar-based AI accelerators. It models the analog VMM signal chain — RRAM cell, access NMOS, BL/SL/WL wire parasitics, boundary drivers, sample-and-hold, mux, and ADC — and lifts the integer output through digital aggregation (shift-add, accumulators) up to a macro layer that exposes one `matmul` per fabricated tile.

## Public surface

The core library's public surface stops at `neurox.architecture`. Above that line — training loops, observer calibration, model rewriting — lives in `example/` and is not a stable API. Below it, every polymorphic family (`CimMacro`, `Solver`, `DiffVadc`, …) follows the same family-base + concrete-config + `from_config` registry pattern documented in [`docs/internals/config_and_policy.md`](docs/internals/config_and_policy.md).

## Layout

- `neurox/primitive/device/` — RRAM, NMOS, Selector device-physics primitives.
- `neurox/primitive/analog/` — VoltageDriver clamp, Vmux, Imux, SwitchCap, and the current-/voltage-domain converter families.
- `neurox/primitive/digital/` — integer accumulators, shift-adders, subtractors.
- `neurox/primitive/xbar/` — `cell/` (abstract `XbarCell` + the `_1t1r` concrete cell), `array/` (abstract `XbarArray` + the `_1t1r` concrete array with its DC solver), and `solver/` (the shared block-tridiagonal DC solver).
- `neurox/primitive/macro/cim/` — abstract `CimMacro`, `IdealCimMacro` reference twin; concrete tiles (e.g. the offset-coded 1T1R tile) live in `works/`.
- `neurox/architecture/unit/` — `UnitBase`/`LinearUnit`/`Conv2dUnit` operator bases with their ideal reference leaves, plus `cim/` (`CimUnit` family with `from_config` factory — the public entry point) and its `engine/` execution pipelines and `slicer/` value-domain slicing primitives.
- `neurox/common/` — `ProfileMixin`/`FabricateMixin`/`RegistryMixin`/`ValidateMixin`, encoding transcoders, `load_dump`, the profiler side-channel, quant primitives.
- `neurox/tools/` — offline calibration / analysis CLIs for cell state maps, ADCs, and solvers.
- `example/` — runnable LeNet and BERT pipelines (float train, HAT, evaluation) showing how a user assembles the above into a training/inference flow.

## Example pipelines

See [`docs/guides/algorithm_engineer/`](docs/guides/algorithm_engineer/README.md) for the LeNet and BERT walkthroughs.

## Developer docs

See [`docs/`](docs/README.md) — the scientific reference, implementation internals, contributing standards, and ADRs.
