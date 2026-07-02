# Digital circuits

The exact integer-datapath primitives stitched around the analog core: each computes a deterministic digital function on integer tensors and carries a behavioural PPA cost model (no gate-level netlist). This layer is where the digit-radix recombination, partial-sum accumulation, and signed add/subtract that surround a VMM are accounted.

- [accumulator](accumulator.md) — modular-arithmetic sum reduction over one integer axis.
- [adder](adder.md) — element-wise integer add.
- [subtractor](subtractor.md) — element-wise integer subtract.
- [shift_adder](shift_adder.md) — radix-weighted positional digit recombination with modular wrap.

The fixed-point multiply-shift requantize step is not modelled as a separate digital block — it lives inline at the consumer's call site.
