# Device

Device-level physics: the smallest electrical primitives, each specifying its current-voltage law, programmable / operating state, non-idealities, and parameters.

- [rram](rram.md) — resistive memory cell: hyperbolic-sine I-V, programmable conductance state, the programming and read non-ideality stack.
- [mosfet](mosfet.md) — EKV-softplus MOSFET primitive: polarity-parameterized NMOS/PMOS continuous I-V with temperature scaling and Pelgrom mismatch.
- [selector](selector.md) — OTS threshold selector: per-cell threshold-voltage map with Gaussian mismatch.
