# 1T1R Xbar Stack

This directory documents the current 1T1R xbar implementation.

Major components:

- `circuit_core.py` — shape-independent physical core
- `newton_raphson_solver.py` — DC solver and boundary coupling
- `offset.py` — offset-coded xbar orchestration

Current layering:

- mapping / grouping semantics live in the xbar layer above the core
- physical devices and peripherals live in the core
- fabricated state stays in the owning leaf modules
- runtime snapshots are sampled per call and threaded through the solver
