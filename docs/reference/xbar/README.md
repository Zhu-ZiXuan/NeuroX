# Crossbar (xbar)

The crossbar is the physical tile that performs one compute-in-memory vector-matrix multiply (VMM): cells whose programmed conductances multiply an input vector and accumulate as analog current, which a readout chain digitizes. This is where analog physics (IR drop, device noise, finite-gain clamps, ADC quantization) enters the simulator.

The layer is an abstract contract, plus topology families (currently 1T1R; 2T1R / 2T2R reserved) and a readout family:

- [base](base.md) — the `Xbar` abstract contract (primitive operation, value domain, output rescale) and the lossless ideal twin, shared by every topology.
- [_1t1r/](_1t1r/README.md) — the 1T1R topology: the shared physical array, its DC solver, and the offset-coded operating xbar.
- [readout/](readout/README.md) — the voltage-domain readout family that operating xbars compose.
