# General DAC

## Design decisions

- **No fabricated mismatch.** This DAC introduces no static per-output mismatch, so `_sample_fabricate_mismatch` is an explicit no-op.
- **Energy fully on `energy_per_op__fJ`.** All per-element switching energy is lumped onto this single config field.

## Contracts & invariants

- **Construction.** `__init__` follows the [DAC base](base.md) keyword signature and registers `code_to_signal` as a non-persistent buffer at the constructor `dtype`.
- **`convert` side effects.** Beyond returning the sampled voltage, `convert` emits per-call dynamic energy and latency through the profiler side channel; the drive-thermal noise is gated by `policy.drive_thermal`.

## Gotchas

- **Double-counting energy.** Leaving `energy_per_op__fJ` non-zero when the switching energy is already booked elsewhere over-counts; zero it deliberately at the composing stage.

---

- **Reference**: [general](../../../../reference/primitive/analog/dac/general.md)
- **Implementation**: `neurox/analog/dac/general.py`
- **Tests**: TODO - name the guarding test
