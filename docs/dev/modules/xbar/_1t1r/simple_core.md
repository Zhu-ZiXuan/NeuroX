# `neurox/xbar/_1t1r/simple_core.py`

## Current status

Placeholder — currently empty.

`simple_core.py` is reserved for a lighter, behavioural 1T1R core that will not run the full DC Newton solver. It is intended for fast sanity checks and large-array smoke tests where the analog non-idealities covered by `CircuitCore1T1R` would dominate the wall-clock budget without materially changing the correctness signal.

The simple core will expose the same core-side interface as `CircuitCore1T1R` so any consuming 1T1R xbar accepts it as a drop-in replacement.

See also:

- `circuit_core.md` — the full physical core
- `solver.md` / `nested_solver.md` / `full_jacobian_solver.md`
- `docs/dev/modules/xbar/ideal.md` — the tile-level lossless reference; conceptually adjacent but a fully different abstraction layer
