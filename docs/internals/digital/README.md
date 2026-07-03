# Digital

How the digital integer-datapath primitives are built. The spec is in [reference/digital](../../reference/digital/README.md); this side covers only what the code cannot tell you.

All four leaves share a thin `DigitalCircuit` base (`neurox/digital/base.py`) that owns no state; it exists only to declare the fabricate hook as one explicit family-wide no-op, since integer-exact logic has no static manufacturing variation to resample.

- [accumulator](accumulator.md) — modular-wrap reduce, serial-op accounting.
- [adder](adder.md) — element-wise add, no-wrap contract.
- [subtractor](subtractor.md) — element-wise subtract (adder sign twin).
- [shift_adder](shift_adder.md) — radix-weight construction, post-wrap partial-sum.
