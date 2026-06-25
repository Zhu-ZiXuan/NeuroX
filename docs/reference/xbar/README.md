# Crossbar (xbar)

The crossbar is the physical tile that performs one compute-in-memory vector-matrix multiply (VMM): cells whose programmed conductances multiply an input vector and accumulate as analog current, which an inline readout chain digitizes. This is where analog physics (IR drop, device noise, finite-gain clamps, ADC quantization) enters the simulator.

The layer is an abstract contract, plus topology families (currently 1T1R; 2T1R / 2T2R reserved):

- [base](base.md) — the `Xbar` abstract contract (primitive operation, value domain, output rescale) and the lossless ideal twin, shared by every topology.
- [cell](cell.md) — the `XbarCell` abstract contract: the topology-agnostic two-terminal cell branch (single condensed current, signed terminal conductances) the array solver sees.
- [cell_1t1r](_1t1r/cell.md) — the concrete 1T1R cell (RRAM in series with an access NMOS, condensed access node), the shared-kernel cell every 1T1R topology reuses.
- [solver](solver.md) — the topology-agnostic SL/BL IR-drop DC solve: a stateless damped-Newton solver over a pluggable cell and two pluggable clamp drivers.
- [core/](_1t1r/README.md) — the pure physical core array (cell array, wire parasitics, solver), the shared infrastructure a scheme xbar builds on. The concrete `Core1T1R` lives here.

Concrete schemes that build a signed VMM on a core — drivers, boundary reference, coding, and readout — are implemented as scheme xbars on top of this layer.
