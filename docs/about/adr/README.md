# Architectural decision records

ADRs record why NeuroX chose a particular design.

Use ADRs for decisions that are:

- cross-cutting
- long-lived
- likely to be questioned again
- chosen from multiple real alternatives, with the trade-off worth recording
- sensitive to numerical correctness or performance edges that the code itself cannot self-document

Module docs describe the current design. ADRs explain why the project chose that design and what alternatives were rejected.

- [ADR-0001](ADR-0001-config-dispatch-and-owned-construction.md) — config dispatch and owned construction
- [ADR-0002](ADR-0002-nmos-is-a-pure-electrical-primitive.md) — NMOS is a pure electrical primitive
- [ADR-0003](ADR-0003-xbar-cell-abstraction-and-single-nested-solver.md) — pluggable xbar cell and the single nested solver
- [ADR-0004](ADR-0004-clamp-driver-protocol-and-generic-solver.md) — clamp-driver role and the topology-agnostic array solver
- [ADR-0005](ADR-0005-mosfet-is-a-polarity-parameterized-electrical-primitive.md) — MOSFET is a polarity-parameterized electrical primitive
