# Architecture

This document describes the software architecture and code organization of NeuroX. After reading it, a developer should know which directory a new class belongs in and what each modeling level is responsible for — whether that class describes a novel RRAM device, a digital-CiM bit-serial adder, a new cell topology, or a complete chip.

## Design Goals

NeuroX is a general CiM simulation platform. The architecture must support:

- **Many computation schemes**: analog crossbar, digital in-memory, hybrid dataflows.
- **Many device technologies**: RRAM, PCM, MRAM, FeFET, Flash, SRAM, capacitors, selectors, transistors.
- **Many cell topologies**: 1T1R, 1S1R, 1T1C, 1T1MTJ, SRAM bitcell, Flash cell, FeFET cell, etc.
- **Many digital circuit styles**: behavioral fixed-function blocks today, gate-level + standard-cell-library synthesis tomorrow, device-level standard-cell modeling as a long-term goal.
- **Hardware-aware training** on the full stack through PyTorch autograd.

To make that possible, we adopt a strict hierarchical separation of concerns: **every level has one job and a narrow interface to the level above it.** A new device model never touches the macro code; a new macro never touches the solver; a new operator never touches the physical array.

The current codebase implements the RRAM analog-crossbar slice of this space. Future contributions fill in the rest without reshaping the hierarchy.

## Modeling Hierarchy

NeuroX organizes code into seven levels, bottom-up from physical devices to neural-network operators. Each level depends only on the levels below it.

```text
operator     Neural-network layer replacement (Linear, Conv2d, ...)
  macro      Logical array macro: tiling, encoding, aggregation, ADC rescale
    xbar     Physical crossbar cell/array: bias coding, broadcast, circuit solver
      analog     Analog peripherals: ADC, DAC, drivers, sense amplifiers
      digital    Digital circuits: accumulator, shift-add, requantizer, gate netlists
        device   Physical devices: RRAM, PCM, MRAM, NMOS, FeFET, selector, wire, ...
          common  Shared utilities: noise models, config I/O, math primitives
```

The hierarchy is intentionally *behaviour-oriented*, not *technology-oriented*. The `device` level holds whatever primitives compose a cell; `xbar` holds whatever cell wiring sits in a crossbar; `analog` and `digital` hold whatever peripherals surround the array. Analog CiM and digital CiM share the same backbone — they differ only in which classes sit at `xbar`, `analog`, and `digital`.

## Level Details

### `neurox/device/` — Physical Device Models

Models individual physical components. Each class describes one device's electrical behavior through its I-V equation (or charge equation, or state-transition rule) and its non-idealities.

A device class knows **nothing** about how it is wired or how a solver will use it — it only exposes a pure physics interface.

#### Current implementations

| Module | Device | Physics |
| --- | --- | --- |
| `rram.py` | Resistive RAM (filamentary) | `i = g * sinh(alpha*v)/alpha`; gamma programming noise; drift; RTN + thermal read noise; stuck-at faults |
| `nmos.py` | NMOS transistor (28 nm-class PDK model) | Deep-triode g_on + SPICE subthreshold g_off compiled from geometry (W, L, A_D, P_D, A_S, P_S) and process constants (μ·Cox, V_th, n_factor, T); 50/50 channel-split capacitances compiled from PDK cap densities; symmetric square-law I-V for peripheral circuits. See [`nmos_model.md`](nmos_model.md) |
| `selector.py` | OTS threshold selector | Threshold voltage with Gaussian mismatch |
| `wire.py` | Interconnect wire | Resistance per unit length |

#### Planned extensions

PCM (two-regime crystalline/amorphous model with retention), MRAM-STT and MRAM-SOT (stochastic switching, read disturb), Flash cell (charge-trap with retention and endurance), FeRAM / FeFET (hysteretic polarization), capacitor (linear and MIM with leakage), PMOS transistor, back-to-back diode selectors. Each new device implements the same narrow interface and composes with existing cells and solvers with zero changes above the device layer.

Device classes expose only their forward equation (e.g., `i__uA(v, g)`). Derivatives are the solver's responsibility — either via autograd (`elementwise_diff`) or via closed-form derivatives written inside the xbar file. This keeps the device contract minimal.

### `neurox/analog/` — Analog Peripheral Circuits

Models the interface between the digital and analog domains and any analog support circuitry.

#### Current implementations

| Module | Circuit | Notes |
| --- | --- | --- |
| `adc.py` | General ADC | Boundary-based quantization; sampling / comparator / drive-thermal noise; `torch.bucketize` to `int16` |
| `dac.py` | General DAC | Code-to-signal lookup table with thermal noise |
| `driver.py` | Constant-voltage driver | Nominal drive value with Gaussian thermal noise |

#### Planned extensions

SAR ADC and flash ADC with cycle-accurate bit-serial behavior; sigma-delta ADC; sense amplifier circuits; current mirrors and replica bias generators; OTAs and transimpedance amplifiers for current-mode readout.

All analog modules report per-operation dynamic energy as a 0-d scalar tensor alongside their functional output.

### `neurox/digital/` — Digital Circuits

Models fixed-function digital logic used for post-processing and data aggregation.

#### Current implementations (behavioral)

| Module | Circuit |
| --- | --- |
| `accumulator.py` | Modular sum along a tile axis |
| `shift_adder.py` | Weighted radix-power sum for digit recombination |
| `subtractor.py` | Sign-split code subtraction |
| `adder.py` | Modular addition |
| `requantizer.py` | Fixed-point rescale: multiply by int32 multiplier then right-shift |

Each behavioral module is parameterized by `bit_width` and a small PPA table (energy / latency / leakage / area per operation).

#### Planned extensions

- **Gate-level + netlist modeling.** The same `digital` package will host circuits built from generic standard-cell primitives (NAND, NOR, DFF, mux, ...) with PPA looked up from a technology library — analogous to how synthesis tools estimate at the gate level.
- **Device-level standard cells.** A research-grade path that builds standard cells directly on top of the NMOS/PMOS device models in `neurox/device/`, so the very same device classes drive both analog and digital simulation. This unifies the modeling stack at the price of runtime cost and is intended for technology exploration, not full-network evaluation.
- **Digital CiM arrays.** Bit-serial or bit-parallel in-memory digital multipliers will live under `neurox/xbar/` (because they *are* crossbars), but will be composed from `digital/` primitives and skip the `analog/` peripherals entirely.

### `neurox/xbar/` — Crossbar Cell and Array

Defines the physical array — its cell topology and its circuit solver. This is where one matrix-vector multiply on one physical tile happens.

A crossbar class combines device models from `device/` into a cell, stacks cells into an array, drives the array with analog peripherals from `analog/` and digital logic from `digital/`, and solves the resulting circuit.

#### Current implementations

| Module | Cell | Solver |
| --- | --- | --- |
| `base.py` | Abstract `Xbar` base, `XbarConfig`, ref-column helpers | (shared) |
| `xbar_1t1r_bias.py` | Bias-coded 1T1R with MOSFET-in-triode access switch (ideal or finite `g_on`) | Fully flat: Padé-closed-form V_X + zero-wire warm start + fixed-depth unrolled outer Newton (see [`solver_1t1r.md`](solver_1t1r.md)); dynamic-energy model documented in [`xbar_1t1r_energy.md`](xbar_1t1r_energy.md) |
| `solver_1t1r.py` | — | Flat finite-wire solver + ideal-wire closed-form fast path |
| `solver.py` | Shared primitives | Thomas tridiagonal, KCL residuals, driver-port currents, autograd-based differentiation |

#### Planned extensions

- **Cell topologies:** 1S1R, 1T1C, 1T1MTJ, SRAM-CiM bitcell, Flash-CiM cell, FeFET-CiM cell. Each is a different combination of device models wired into a new `Xbar*` subclass; the solver machinery (Newton loops, Thomas solve, KCL residual) is shared through `solver.py`.
- **Digital CiM arrays:** bit-serial multipliers, charge-domain computing, capacitive in-memory MAC. These reuse the `Xbar` interface but plug into digital (not analog) peripherals.

Each xbar subclass owns its own solver and any helpers that contain cell-specific equations. `solver.py` contains only device-agnostic primitives.

### `neurox/macro/` — Logical Macro

Bridges the gap between a physical tile — which handles one `[col_num, row_num]` matmul — and a full `[N, K]` weight matrix.

A macro owns:

- **Tiling.** Split `[N, K]` into `Tc * Tr` physical tiles with padding.
- **Radix digit encoding.** Decompose signed weights into `Sw` digits and activations into `Sa` digits using true-form, radix-complement, or CSD encoding.
- **Digital aggregation.** Reduce over `Sa`, `Sw`, `Tc`, then combine `(Tr, row_size)` into `N` and add bias. The reduction is fused under `@torch.compile`.
- **ADC rescale correction.** Fold the ADC-range-to-ideal-range ratio into the requantizer's fixed-point `(multiplier, rshift)` pair.

#### Tensor layout convention

```text
[*batch, M, Tc, Tr, Sa, Sw, sign, row_size, col_size]
 ^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 leading                trailing
 (layer-varying)        (config-fixed: digits, sign, cell geometry)
```

This split is not specific to RRAM — it is shared by every planned CiM scheme. Analog and digital CiM both produce per-cell partial results that share the trailing layout; only the `xbar` level differs.

### `neurox/operator/` — Neural-Network Operators

Replaces standard PyTorch layers with hardware-aware versions that delegate the integer matmul to a macro.

| Module | Role |
| --- | --- |
| `base.py` | `NeuroxOperator` (inference) and `NeuroxQAT` (training mixin with EMA ranges, fake-quantization, STE backward) |
| `linear.py` | `QuantLinear` and `LinearQAT` for `nn.Linear` |
| `conv2d.py` | `QuantConv2d` and `Conv2dQAT` for `nn.Conv2d` (im2col, grouped matmul) |
| `qat_util.py` | Fixed-point scale derivation: float scale to `(int32 multiplier, int32 right-shift)` |

Planned extensions include automatic graph rewriting via `torch.fx` so users do not have to call `replace_qat` manually, plus operator variants for attention, matmul, and other non-Linear / non-Conv2d primitives.

### Top-level user-facing API

The user-facing API is exposed at the top of the package (`import neurox`)
and is organised around three lifecycle stages:

1. **Preparation / training-side conversion** — `replace_for_hat`,
   `freeze_hat_observers`, `fold_batchnorm`, `extract_neurox_state`,
   `QuantSpec`.  Converts a float model into NeuroX-aware QAT form
   and exports a NeuroX-flat checkpoint.
2. **Evaluator construction** — `build_evaluator` (one-shot wrapper)
   plus the four staged building blocks `replace_model`,
   `load_neurox_state`, `bind_output_calibration`, `fabricate_model`,
   and the rule-driven policy types `ReplacementPolicy`,
   `ReplacementRule`, `ReplacementContext`, `default_policy`, helper
   constructors (`name_excluded_match`, `by_attr_match`,
   `heterogeneous_macro_policy`), plus the state diagnostics
   `NeuroxStateError`, `StateBindingReport`, `StructuralReport`.
3. **Execution + profiling** — `NeuroxProfiler` (context manager),
   `ProfilerReport`, `StaticMetrics`, `RuntimeEvent`, `StaticRecord`,
   and the `ProfiledModule` mixin used internally by physical
   modules.  Numerical functions return clean numerical results;
   every physical leaf emits dynamic energy + latency through the
   profiler side channel; leakage *energy* is derived centrally in
   `NeuroxProfiler.summary` from `static.leakage_power__uW *
   total_latency__ns`.

| Submodule | Role |
| --- | --- |
| `neurox/replace/` | All structural-replacement + state-loading code; submodules `policy.py`, `state.py`, `report.py`, `fold.py`. |
| `neurox/profiler/` | `NeuroxProfiler` + `ProfiledModule` side-channel mixin. |

### `neurox/common/` — Shared Utilities

| Module | Role |
| --- | --- |
| `noise.py` | Noise config NamedTuples and `apply_*` helpers (Gaussian, telegraph, gamma, lognormal, stuck-at); fused under `@torch.compile` |
| `config.py` | `Config` base class with JSON/YAML I/O and `ConfigBuilder` pattern |
| `load_dump.py` | Dataclass serialization helpers |

### `neurox/config/` — Default Configurations

Ready-to-use parameter sets (e.g., `default_1t1r.py`) that wire together device, circuit, digital, and macro configs for a complete, runnable system. As the device and cell zoo grows, more default configurations will be added here.

## Code Organization Rules

1. **One primary class per file.** Supporting types (configs, NamedTuples, private helpers) may share the file.
2. **Configs co-locate with their consumer** unless shared across levels.
3. **Subpackage `__init__.py` exports the public API.** External code imports from the package, not from private modules.
4. **Filenames use `snake_case`** matching the primary class name.
5. **New technologies slot into existing levels.** A new device becomes a file in `neurox/device/`. A new cell topology becomes an `Xbar*` subclass in `neurox/xbar/`. A new digital block becomes a module in `neurox/digital/`. The hierarchy does not expand sideways — it deepens.
