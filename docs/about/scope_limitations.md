# Scope and limitations

The global fidelity envelope of the NeuroX simulator: what it does and does not model, the validity ranges it operates within, and the simplifying assumptions it makes across all subsystems.

It is an aggregate, not the source of truth. Every subsystem's own boundary lives in the Assumptions, scope and validity section of its [Reference](../reference/README.md) document; this page collects those per-model statements into one envelope, so a reader can judge end to end whether an experiment is inside the simulator's intended regime. When a subsystem entry and its Reference document disagree, the Reference document wins and this page is the one to correct.

Read an entry here as a claim about the whole pipeline: a limitation local to one device, such as an unimplemented retention model, becomes a limitation of every macro built on that device, and this page exists to make that propagation visible.

## Per-area envelope

Each area's domain author owns its entry and keeps it in sync with the corresponding Reference Assumptions section. An entry states what is modeled, what is deliberately not modeled, the validity range, and the simplifying assumptions taken.

### Devices

TODO — aggregate the Assumptions of each device Reference document (RRAM, MOSFET, selector).

### Crossbar and solvers

TODO — aggregate the Assumptions of the xbar Reference documents (cell array, IR-drop solver, offset operation).

### Analog peripherals

TODO — aggregate the Assumptions of the analog Reference documents (ADC, DAC, driver).

### Digital circuits

TODO — aggregate the Assumptions of the digital Reference documents.

### Macro

TODO — aggregate the Assumptions of the macro Reference documents (matrix tiling and precision slicing).

### Profiling and PPA

TODO — state the fidelity boundary of the energy, power, area, and latency estimates.

### Chip-level and system

TODO — state that multi-macro, NoC, off-chip memory, and full-system modeling are not in scope.
