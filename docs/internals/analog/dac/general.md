# General DAC

## Summary

`GeneralDAC` (`dac/general.py`) is the LUT DAC: a code → voltage table plus optional additive Gaussian drive-thermal noise.

## Design decisions

- **No fabricated mismatch.** The only non-ideality (drive-thermal) is dynamic, applied inside `convert`; `_sample_fabricate_mismatch` is an explicit no-op. `fabricate()` resolves only the profiler-inst tally locked at `__init__`.
- **Energy fully on `energy_per_op__fJ`.** All per-element switching energy is lumped onto this single config field, zeroed when the same energy is booked at another stage.

## Contracts & invariants

- **`convert(code)`** indexes `code_to_signal`, applies `drive_thermal__V` gated by `policy.drive_thermal`, and emits per-call dynamic energy through the profiler side channel.
- **`__init__`** builds the `code_to_signal` LUT buffer from `config`; the shared construction signature is the [DAC base](base.md) contract.

## Performance & resources

N/A - a single LUT index per call.

## Gotchas

- **Double-counting energy.** Leaving `energy_per_op__fJ` non-zero when the switching energy is already booked elsewhere over-counts; zero it deliberately at the composing stage.

## Known limitations

- N/A.

---

- **Reference**: [general](../../../reference/analog/dac/general.md)
- **Implementation**: `neurox/analog/dac/general.py`
- **Tests**: TODO - name the guarding test
