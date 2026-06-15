# Scope and Limitations

This document is the global fidelity envelope of the NeuroX simulator: a single place that states what NeuroX does and does not model, the validity ranges it operates within, and the simplifying assumptions it makes across all subsystems.

It is an aggregate, not the source of truth. Every subsystem's own boundary lives in the Assumptions, scope and validity section of its [Reference](../reference/README.md) document; this page collects those per-model statements into one envelope so a reader can judge, end to end, whether a given experiment is inside the simulator's intended regime. When a subsystem entry and its Reference document disagree, the Reference document wins and this page is the one to correct.

Read an entry here as a claim about the whole pipeline: a limitation that is local to one device (say, a retention model that is not implemented) becomes a limitation of every macro built on that device, and that propagation is what this page is meant to make visible.

## Per-area envelope

The entries below are owned by each area's domain author and must be kept in sync with the corresponding Reference Assumptions section. Each entry should state what is modeled, what is deliberately not modeled, the validity range, and the simplifying assumptions taken.

### Devices

TODO — aggregate the Assumptions of each device Reference document (RRAM, NMOS, selector).

### Crossbar and solvers

TODO — aggregate the Assumptions of the xbar Reference documents (cell array, IR-drop solver, offset operation).

### Analog peripherals

TODO — aggregate the Assumptions of the analog Reference documents (ADC, DAC, driver).

### Digital circuits

TODO — aggregate the Assumptions of the digital Reference documents.

### Macro and mapping

TODO — aggregate the Assumptions of the macro and mapper Reference documents.

### Profiling and PPA

TODO — state the fidelity boundary of the energy, power, area, and latency estimates.

### Chip-level and system

TODO — state that multi-macro, NoC, off-chip memory, and full-system modeling are not yet in scope (see the [roadmap](roadmap.md)).
