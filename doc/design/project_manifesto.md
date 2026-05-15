# Project Manifesto

## Vision

**NeuroX aims to be a general, end-to-end benchmarking platform for compute-in-memory (CiM) systems**, bridging algorithm, architecture, circuit, and device research within a single PyTorch-native framework. We want one tool that answers three classes of question at once:

- **Accuracy:** How does this neural network behave on this hardware, with this training recipe?
- **Energy:** How many joules does one inference cost, broken down by layer, circuit, and device?
- **Circuit PPA:** What is the area, leakage, and latency of the full chip that runs this workload?

CiM is not a single architecture. It spans analog and digital computation, multiple memory technologies, and countless cell topologies. A useful platform must embrace this diversity from day one — not bake in assumptions about any single style.

## Why NeuroX Exists

CiM simulation today is fragmented across disconnected tools:

- **SPICE-level simulators** capture device physics and circuit nonlinearity accurately, but cannot run a full neural network, and are disconnected from deep-learning frameworks.
- **Architecture-level simulators** (e.g., NeuroSim) estimate PPA but are typically written in C++, tied to one cell type, and disconnected from the training loop — hardware-aware training is impractical.
- **Algorithm-level noise injectors** add generic Gaussian noise to weights but ignore the actual circuit topology — IR drop, nonlinear I-V, selector threshold variation, telegraph noise, ADC clipping, and their interactions.

None of these tools give an algorithm researcher, an architecture explorer, and a device engineer a shared workbench. NeuroX is built to be that workbench — accurate enough to capture the dominant non-idealities, configurable enough to describe new devices and new architectures, and fast enough to evaluate full networks on a single GPU.

## Scope

NeuroX targets the full CiM design space. The list below shows the intended scope; the current codebase implements the highlighted starting point.

### Computation schemes

- **Analog CiM** — matrix-vector multiply in the analog domain via crossbar arrays. *(currently supported: RRAM analog crossbar)*
- **Digital CiM** — bit-parallel or bit-serial in-memory logic. *(planned)*
- **Hybrid analog-digital dataflows** for mixed-precision or partial-sum aggregation. *(planned)*

### Memory technologies (device layer)

- **Resistive:** RRAM (filamentary and interfacial), PCM, FeRAM
- **Magnetic:** STT-MRAM, SOT-MRAM
- **Charge-based:** SRAM, capacitor-based, Flash (NOR/NAND), FeFET
- **Selectors:** OTS, threshold-switching selectors, back-to-back diodes

Current code implements an RRAM filament model with the sinh I-V nonlinearity plus an NMOS-selector model; every other entry above is a planned extension that will share the same `Device` contract.

### Cell topologies

- **1T1R, 1S1R, 1T1C, 1T1MTJ, SRAM bitcell, Flash cell, FeFET** — each is a specific combination of devices wired into a crossbar or array. Every cell type plugs into the same `Xbar` interface so higher-level macros and operators do not need to know which technology they are running on.

Current code implements 1T1R with an ideal switch and with an NMOS selector.

### Digital circuit modeling

- **Behavioral digital modules** — accumulator, shift-adder, requantizer, etc., characterized by bit-width and PPA numbers. *(currently supported)*
- **Gate-level + netlist** — build digital circuits from standard-cell libraries, similar to how synthesis tools estimate PPA against a tech lib. *(long-term)*
- **Standard-cell synthesis from NMOS/PMOS device models** — build logic gates on top of the same device classes used for analog simulation. *(long-term research goal)*

The long-term digital roadmap lets the same framework evaluate both analog and digital CiM within one consistent abstraction stack.

### Algorithm and training integration

- **PyTorch-native inference** — drop-in replacement for `nn.Linear` / `nn.Conv2d`. *(currently supported)*
- **Quantization-aware training (QAT)** with straight-through estimators through the full analog-digital pipeline. *(currently supported)*
- **Hardware-aware training** — gradients flow through programming noise, read noise, IR drop, and ADC quantization.
- **Automatic graph transformation** via `torch.fx` for zero-code-change model conversion. *(planned)*
- **Neural architecture search** under hardware constraints. *(planned)*

## Who Should Use NeuroX

### Algorithm researchers

You want to know how your Transformer, CNN, or diffusion model behaves on CiM hardware, and whether hardware-aware training recovers the accuracy gap. NeuroX lets you replace standard PyTorch layers with hardware-aware versions via a single API call. Training and inference remain standard PyTorch — `loss.backward()` flows gradients through the quantized, noisy analog pipeline.

### Architecture explorers

You want to compare tile sizes, ADC resolutions, mapping policies, or bit-serial vs. bit-parallel digital aggregation. Every parameter lives in a frozen dataclass config; change a number and re-run. The profiler gives you per-layer energy and area so you can chase efficiency bottlenecks across the full stack.

### Circuit and device designers

You have a new RRAM process, a novel ADC, a SOT-MRAM cell, or an experimental selector. NeuroX's modular architecture lets you plug in a new device, circuit, or cell model by implementing a small interface (forward I-V, noise sources, PPA numbers). Your model immediately composes with every higher level — full networks, training loops, chip-level simulations — with no extra integration work.

## Core Principles

1. **Physical quantities everywhere.** Voltage is in volts, current in milliamps, conductance in millisiemens, energy in nanojoules, area in square micrometers. No unitless normalizations that hide modeling errors.

2. **Hierarchical, interchangeable modeling.** The stack is cleanly layered: device → circuit (analog / digital) → cell → array → macro → operator. Every level has a narrow interface, so a new device does not require changes above it and a new macro does not require changes below it.

3. **Configuration-driven.** All device, circuit, and macro parameters live in frozen dataclass configs. No magic constants buried in source code. The same model description can be shared, versioned, and swapped between experiments.

4. **PyTorch-native and differentiable.** Every tensor operation goes through PyTorch; the analog pipeline is autograd-compatible through straight-through estimators. `torch.compile` is the primary performance tool — code is written to trace cleanly under TorchInductor.

5. **Separation of physics and orchestration.** Device classes describe I-V and noise; solvers compute operating points; macros handle tiling and encoding; operators handle quantization and model integration. Each level does one thing well.

6. **Open and community-driven.** NeuroX is intended for the research community. Contributions that extend the device zoo, add new circuit models, or support new CiM schemes are welcomed as first-class citizens, not afterthoughts.
