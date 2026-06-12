# Roadmap

NeuroX is a general CiM simulation and benchmarking platform. This document tracks what is available today and what is planned for the future, organized by concern. Items marked `[x]` are implemented; `[ ]` items are on the roadmap.

## Computation Schemes

- [x] Analog crossbar CiM with RRAM cells
- [ ] Digital CiM arrays (bit-serial and bit-parallel in-memory multipliers)
- [ ] Charge-domain CiM (capacitor-based accumulation)
- [ ] Hybrid analog-digital dataflows (e.g., analog MAC + digital partial-sum aggregation)

## Device Models (`neurox/device/`)

### Resistive memory

- [x] RRAM filament model (sinh I-V, programming gamma noise, power-law drift, RTN + thermal read noise, stuck-at faults)
- [ ] PCM with crystalline/amorphous two-regime model, retention, and resistance drift
- [ ] FeRAM with hysteretic polarization switching

### Magnetic memory

- [ ] STT-MRAM with stochastic switching and read disturb
- [ ] SOT-MRAM with three-terminal write
- [ ] MTJ I-V including TMR ratio and bias dependence

### Charge-based memory

- [ ] SRAM bitcell (6T / 8T)
- [ ] Flash cell (NOR and NAND) with charge trapping, retention, endurance
- [ ] FeFET with polarization-dependent threshold
- [ ] Capacitor (linear, MIM) with leakage

### Transistors and selectors

- [x] NMOS transistor (symmetric square-law, threshold mismatch)
- [x] OTS threshold selector with Gaussian mismatch
- [ ] PMOS transistor model
- [ ] Back-to-back diode selector
- [ ] Ovonic / chalcogenide selector with bi-directional threshold

### Interconnect

- [x] Per-line segment R / C carried as core-config scalars (BL/SL/WL, first-segment + cell-to-cell)
- [ ] Distributed RC wire model with on-cell parasitic capacitance

## Crossbar Cells and Arrays (`neurox/xbar/`)

- [x] 1T1R with ideal (always-on) access switch
- [x] 1T1R with NMOS access transistor
- [ ] 1S1R (selector + RRAM)
- [ ] 1T1C (DRAM-like CiM bitcell)
- [ ] 1T1MTJ (MRAM-based CiM)
- [ ] SRAM-CiM (6T / 8T with CiM peripherals)
- [ ] Flash-CiM (NOR-Flash analog MAC)
- [ ] FeFET-CiM
- [ ] Multi-level cell (MLC) variants for any of the above

## Circuit Solvers (`neurox/xbar/solver.py`)

- [x] Damped Newton-Raphson IR-drop solver
- [x] Thomas-algorithm tridiagonal solve
- [x] Schur-complement solver for three-terminal access devices
- [x] Autograd-based device differentiation (`elementwise_diff`)
- [x] Closed-form differential conductance for sinh I-V
- [ ] Modified nodal analysis (MNA) for general topologies
- [ ] Transient solver for switching-dependent phenomena

## Analog Peripherals (`neurox/analog/`)

- [x] General boundary-based ADC with sampling, comparator, and drive thermal noise
- [x] General DAC with code-to-signal lookup and thermal noise
- [x] Constant-voltage driver
- [ ] SAR ADC with cycle-accurate bit-serial behavior
- [ ] Flash ADC
- [ ] Sigma-delta ADC
- [ ] Sense amplifier (current, voltage, and latched types)
- [ ] Current mirror and replica bias
- [ ] OTA and transimpedance amplifier

## Digital Circuits (`neurox/digital/`)

### Behavioral (current)

- [x] Accumulator, adder, subtractor, shift-adder
- [x] Modular-arithmetic with configurable bit-width
- [x] `@torch.compile` fused digital aggregation pipeline

### Gate-level and netlist (planned)

- [ ] Standard-cell library loader (timing, power, area per cell from a tech-lib)
- [ ] Gate-level netlist simulator (NAND, NOR, DFF, MUX, ADD primitives)
- [ ] Post-synthesis PPA estimation based on gate counts and critical paths
- [ ] Integration with open-source cell libraries (SkyWater 130nm, ASAP7)

### Device-level standard cells (long-term research)

- [ ] Build NAND/NOR gates from `device/nmos.py` + a future `device/pmos.py`
- [ ] Derive standard-cell PPA from device models so the same transistor models drive both analog and digital simulation
- [ ] Technology exploration flow: sweep a device parameter, see both analog accuracy and digital logic PPA change

## Macro and Aggregation (`neurox/macro/`)

- [x] Radix digit encoding (true-form, radix complement, CSD)
- [x] Weight and activation tiling with automatic padding
- [x] Tensor layout `[*batch, M, Tc, Tr, Sa, Sw, sign, row, col]`
- [ ] Leading-dimension merge and serialization (`sim_chunk_size`)
- [x] Compiled digital aggregation under `@torch.compile`
- [x] ADC rescale correction folded into the operator-side fixed-point requantize
- [ ] Per-layer fine-grained mapping policy (weight-stationary, output-stationary, transposed)
- [ ] Heterogeneous macro support (mixed precision, mixed technology)

## Operator and Model Integration (out of core)

Operator wrapping and model rewriting are application-layer concerns; the public NeuroX surface stops at `neurox.macro`. `example/lenet/` and `example/bert/` ship reference implementations of QAT-aware quantised layers, observer calibration, and the train-float → HAT → evaluate flow. Items below are tracked only as possible future *core helpers* that would simplify those examples, not as core-API additions:

- [ ] Reference QAT/observer utility module (extracted from `example/*/quant.py` if multiple examples need it).
- [ ] `torch.fx`-based model rewriting helper.
- [ ] Attention / grouped-conv mapping helpers.
- [ ] Hardware-aware training with full analog gradient flow (not just STE).
- [ ] NAS integration hooks.

## Chip-Level and System Modeling

- [ ] Multi-macro chip with global buffer, memory hierarchy, and data router
- [ ] Network-on-chip (NoC) modeling with per-packet energy and latency
- [ ] Off-chip DRAM access modeling for weight and activation streaming
- [ ] Pipeline and dataflow scheduling
- [ ] Full-system energy and throughput reports

## Profiling and Benchmarking

- [x] `NeuroxProfiler`: per-layer dynamic energy and area profiling via context manager
- [x] Static PPA aggregates (area, leakage power, leakage energy, latency)
- [ ] CSV / JSON export of detailed PPA reports
- [ ] Built-in benchmark suite (ImageNet models, Transformers, diffusion models)
- [ ] Accuracy vs. energy vs. area Pareto-front plotting
- [ ] Regression tracking across commits

## Tooling and Infrastructure

- [x] `ruff` formatter and linter
- [x] `mypy` type checking
- [x] Google-style docstrings
- [x] MkDocs + `mkdocstrings` documentation
- [ ] Comprehensive unit and integration test suite with high coverage
- [ ] Continuous integration (GitHub Actions) with GPU smoke tests
- [ ] Pre-built container images (runtime and development)
- [ ] Configuration schema validation with JSON Schema

## Performance Optimization

- [x] `torch.compile` fusion on noise, Newton residual, Newton step, driver currents, digital aggregation
- [x] bfloat16 analog buffers
- [ ] Leading-dimension serialization with `sim_chunk_size`
- [x] Per-call broadcast-shape read noise
- [ ] CUDA-graph capture for repeated inference
- [ ] Multi-GPU and distributed simulation for large chips
- [ ] Remove the temporary `@torch.compiler.disable` on `Offset1T1RXbar.vec_mat_mul` (`compile_policy.md`): requires rewriting `McsSarAdc.convert`'s SAR bit-loop to a graph-friendly form so inductor compile time drops back to seconds.
