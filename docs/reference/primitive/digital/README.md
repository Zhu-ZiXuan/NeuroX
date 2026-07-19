# Digital circuits

The exact integer-datapath primitives: each computes a deterministic digital function on integer tensors and is modelled behaviourally, with no gate-level netlist. Each document specifies the block's integer function and its PPA cost model.

- [accumulator](accumulator.md) — modular-arithmetic sum reduction over one integer axis.
- [serial_accumulator](serial_accumulator.md) — the same reduction over a time-serial axis, billed per arriving input element.
- [adder](adder.md) — element-wise integer add.
- [subtractor](subtractor.md) — element-wise integer subtract.
- [shift_adder](shift_adder.md) — radix-weighted positional digit recombination with modular wrap.
