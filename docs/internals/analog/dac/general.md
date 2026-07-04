# General DAC

## Summary

`GeneralDAC` (`dac/general.py`) is the LUT DAC: a code → voltage table plus optional additive Gaussian drive-thermal noise. Spec: [reference/analog/dac/general](../../../reference/analog/dac/general.md).

## Design decisions

- **No fabricated mismatch.** The only non-ideality (drive-thermal) is dynamic, applied inside `convert`; `_sample_fabricate_mismatch` is an explicit no-op. `fabricate()` resolves only the profiler-inst tally locked at `__init__`.
- **Energy fully on `energy_per_op__fJ`.** The per-element switching energy is one config field; set it to zero whenever the same energy is accounted at another stage, to avoid double-counting across the readout chain.

## Contracts & invariants

- **`convert(code)`** indexes `code_to_signal`, applies `drive_thermal__V` gated by `policy.drive_thermal`, and emits per-call dynamic energy through the profiler side channel.
- **`__init__(*, config, policy, name, inst_shape, dtype, T__K)`** builds the `code_to_signal` LUT buffer, forwards `name` / `inst_shape` to the base, and records the instance count for the profiler.

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
- **Decisions**: N/A — no ADR governs this module.
