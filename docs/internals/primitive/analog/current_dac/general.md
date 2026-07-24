# General current DAC

## Design decisions

- **No fabricated mismatch.** This DAC introduces no static per-output mismatch, so `_sample_fabricate_mismatch` is an explicit no-op.
- **Energy fully on `energy_per_op__fJ`.** All per-element switching energy is lumped onto this single config field.

## Contracts & invariants

- **Construction.** `__init__` follows the [current DAC base](base.md) keyword signature and registers `_code_to_signal` as a non-persistent buffer at the constructor `dtype`.
- **`convert` side effects.** Beyond returning the sampled current, `convert` emits per-call dynamic energy and latency through the profiler side channel; the drive-thermal noise is gated by `policy.drive_thermal`.

## Gotchas

- **Double-counting energy.** Leaving `energy_per_op__fJ` non-zero when the switching energy is already booked elsewhere over-counts; zero it deliberately at the composing stage.

## Known limitations

- **Output noise is signal-independent.** `drive_thermal__uA` applies one constant additive Gaussian sigma at every code. A real current-steering DAC's dominant output noise is code-dependent — the shot and thermal noise of the steered current sources grows with the steered current ($\sigma \propto \sqrt{I}$ in the shot-noise limit) — so the constant-sigma model over-states noise near zero code and under-states it at full scale. TODO (domain author): specify the code-dependent output-noise law and its parameterisation.
- **Unit-source mismatch is absent.** The LUT is exact at every fabricated instance, so the unit-current-source mismatch that sets a real current-steering DAC's INL and DNL is not expressed. TODO (domain author): specify the unit-source mismatch model and its integral / differential nonlinearity consequence.

---

- **Reference**: [general](../../../../reference/primitive/analog/current_dac/general.md)
- **Implementation**: `neurox/primitive/analog/current_dac/general.py`
- **Tests**: TODO - name the guarding test
