# Digital

How the digital integer-datapath primitives are built. The spec is in [reference/digital](../../reference/digital/README.md); this side covers only what the code cannot tell you.

- [accumulator](accumulator.md) — modular-wrap reduce, serial-op accounting.
- [adder](adder.md) — element-wise add, no-wrap contract.
- [subtractor](subtractor.md) — element-wise subtract (adder sign twin).
- [shift_adder](shift_adder.md) — radix-weight construction, post-wrap partial-sum.
