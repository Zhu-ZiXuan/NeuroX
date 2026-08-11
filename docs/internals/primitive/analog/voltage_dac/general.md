# General voltage DAC

## Design decisions

- **No fabricated mismatch.** This DAC introduces no static per-output mismatch, so `_sample_fabricate_mismatch` is an explicit no-op.
- **Energy is a per-code lookup parallel to the signal lookup.** `code_to_per_op_energy__fJ` is indexed by the same code as `code_to_signal` and states what converting ONE element to that level costs, so each level answers for its own drive event instead of a single lump standing for all of them. The two tables are validated to the same length, and an entry may legitimately be zero — a level whose drive event costs nothing, or one whose cost is booked by another block, is stated as zero rather than approximated by a shared average.

## Contracts & invariants

- **Construction.** `__init__` follows the [voltage DAC base](base.md) keyword signature and registers both lookups as non-persistent buffers: `_code_to_signal` at the constructor `dtype`, `_code_to_per_op_energy__fJ` at the energy dtype, which is the accounting domain's rather than the signal's.
- **`convert` side effects.** Beyond returning the sampled voltage, `convert` emits per-call dynamic energy through the profiler side channel; the drive-thermal noise is gated by `policy.drive_thermal`.
- **The energy payload is gathered, not expanded.** Each element costs what its own code costs, so the emission is the energy LUT indexed by the same code tensor the signal LUT was — one value per converted element, at the code's own layout. `inst_shape` never reaches the forward path: it sizes the static PPA and no per-instance buffer multiplies the signal, so the payload has no instance axis to sum.

## Gotchas

- **Double-counting energy.** Leaving a code's entry non-zero when that level's drive event is already booked elsewhere over-counts; zero those entries deliberately at the composing stage, entry by entry rather than for the whole table.

---

- **Reference**: [general](../../../../reference/primitive/analog/voltage_dac/general.md)
- **Implementation**: `neurox/primitive/analog/voltage_dac/general.py`
- **Tests**: TODO - name the guarding test
