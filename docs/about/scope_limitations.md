# Scope and limitations

The simulator's fidelity envelope follows each model's Assumptions, scope and validity section in [Reference](../reference/README.md). A composite inherits its components' limitations. Domain authors maintain the aggregate below against those authoritative model statements.

## Per-area envelope

Each entry must identify modeled effects, deliberate omissions, validity ranges, and simplifying assumptions.

- **Devices** — TODO: aggregate the device-model assumptions.
- **Crossbar and solvers** — TODO: aggregate cell, array, IR-drop solver, and offset-operation assumptions.
- **Analog peripherals** — TODO: aggregate converter and driver assumptions.
- **Digital circuits** — TODO: aggregate arithmetic and characterization assumptions.
- **Macro** — TODO: aggregate mapping and precision-slicing assumptions.
- **Profiling and PPA** — TODO: state the fidelity boundary of energy, power, area, and latency estimates.
- **Chip and system** — TODO: state the scope of multi-macro, NoC, off-chip memory, and full-system modeling.
