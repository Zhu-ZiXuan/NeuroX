# Digital

How the digital integer-datapath primitives are built.

- [base](base.md) — the `DigitalBase` family base (PPA + `DigitalConfig` / `DigitalPolicy`) and its fabricate no-op.
- [accumulator](accumulator.md) — modular-wrap reduce, serial-op accounting.
- [serial_accumulator](serial_accumulator.md) — time-serial reduce twin, per-input billing.
- [adder](adder.md) — element-wise add, no-wrap contract.
- [subtractor](subtractor.md) — element-wise subtract (adder sign twin).
- [shift_adder](shift_adder.md) — radix-weight construction, post-wrap partial-sum.
