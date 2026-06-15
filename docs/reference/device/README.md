# Device

Device-level physics: the smallest electrical primitives a crossbar cell is built from, each specifying its current-voltage law, programmable / operating state, non-idealities, and parameters. A device owns local physics and fabricated state only; mapping, array geometry, and circuit orchestration live above it.

- [rram](rram.md) — resistive memory cell: hyperbolic-sine I-V, programmable conductance state, the programming and read non-ideality stack.
- [nmos](nmos.md) — EKV-softplus NMOS transistor primitive: continuous I-V with temperature scaling and Pelgrom mismatch.
- [selector](selector.md) — OTS threshold selector: per-cell threshold-voltage map with Gaussian mismatch.
