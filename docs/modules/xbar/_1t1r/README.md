# 1T1R Xbar Stack

This directory documents the current 1T1R xbar implementation.

Major components:

- `circuit_core.py` — shape-independent physical core
- `solver.py` — 1T1R-family solver base + shared DCOP / Residuals containers
- `nested_solver.py` — block-Gauss-Seidel + nested V_clamp Newton (chip-preset default)
- `full_jacobian_solver.py` — fully-coupled Newton with V_X in the global unknown vector
- `offset.py` — offset-coded xbar orchestration

Current layering:

- mapping / grouping semantics live in the xbar layer above the core
- physical devices and peripherals live in the core
- fabricated state stays in the owning leaf modules
- runtime snapshots are sampled per call and threaded through the solver
- solvers are plain stateless tool classes bound to the core; they are not part of the core's PyTorch module tree and carry no buffers or registered state
