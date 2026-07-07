# Selector

## Summary

`Selector` (`selector.py`) is a thin stateful module: it holds a per-cell threshold map, resamples it at fabricate time, and broadcasts it to a reference tensor on request. This document covers only the non-obvious choices.

## Design decisions

- **`inst_shape` is an `__init__` argument, not a config field.** The per-instance shape is a property of the deployment instance, not of the device, so it is passed to the constructor and kept out of the frozen `SelectorConfig`. The config carries the nominal threshold and the mismatch sigma only.
- **Mismatch sigma stays in the config; the policy is the on/off switch.** Unlike MOSFET, whose area-scaled sigma is derived from a matching coefficient, the selector keeps the additive sigma `vth_mismatch__V` directly in `SelectorConfig` (it is the device's intrinsic spread); `SelectorPolicy.vth_mismatch` only gates whether the sample is applied. With the flag off the threshold map is the uniform nominal.
- **`T__K` is stored but unused.** The constructor takes the uniform device-construction context (`dtype`, `T__K`) for interface uniformity with the other devices; the current selector model has no temperature dependence, so `T__K` is held, not consumed.

## Contracts & invariants

- **`_sample_fabricate_mismatch()` resamples the full map per call.** Each `fabricate()` re-expands `nominal_vth__V` to `(*self._inst_shape,)` and resamples; the owning module drives the cadence via `fabricate()` and the mixin auto-cascades.
- **`sample_vth_like(reference)` is the read path.** It broadcasts the fabricated `vth__V` to the reference tensor's shape / device / dtype. The reference tensor defines the target — the selector imposes no shape of its own.

## Performance & resources

State is one threshold buffer at `inst_shape` plus its scalar nominal. `sample_vth_like` broadcasts (no copy of the underlying storage) to the reference shape.

## Gotchas

- **The selector publishes a threshold, not a conduction current.** It owns no I-V law; a consumer that needs switching behavior must build it on top of the threshold map. Do not treat the selector as a current source.

## Known limitations

- No test coverage. `Selector` has no dedicated test module and is not exercised by the 1T1R physics path. A focused unit test (threshold sampling, mismatch statistics, broadcast shape) is a coverage gap.

---

- **Reference**: [selector](../../reference/device/selector.md)
- **Implementation**: `neurox/device/selector.py`
- **Tests**: TODO — no dedicated selector tests yet
