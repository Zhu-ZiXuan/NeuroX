# Crossbar (xbar)

The crossbar is the physical tile that performs one compute-in-memory vector-matrix multiply (VMM): cells whose programmed conductances multiply an input vector and accumulate as analog current, which an inline readout chain digitizes. This is where analog physics (IR drop, device noise, finite-gain clamps, ADC quantization) enters the simulator.

The array layer is a topology-agnostic abstract contract plus the concrete 1T1R topology:

- [cell](cell/README.md) — the `XbarCell` abstract contract: the topology-agnostic two-terminal cell branch (single condensed current, signed terminal conductances) the array solver sees.
- [cell_1t1r](cell/_1t1r/cell.md) — the concrete 1T1R cell (RRAM in series with an access NMOS, condensed access node), the shared-kernel cell every 1T1R topology reuses.
- [array](array/_1t1r/array.md) — the pure physical array (cell array, wire parasitics, solver), the shared infrastructure a scheme xbar builds on. The concrete `XbarArray1t1r` lives here.
- [solver](solver/README.md) — the topology-agnostic SL/BL IR-drop DC solve: a damped-Newton solver over a pluggable cell and two pluggable clamp drivers.

The tile-level primitive operation, value domain, output-rescale grid, and the lossless ideal twin live in [macro/cim](../macro/cim/README.md). Concrete schemes that build a signed VMM on a core — drivers, boundary reference, coding, and readout — are implemented as scheme xbars on top of this layer.
