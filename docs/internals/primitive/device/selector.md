# Selector

`Selector` is a thin stateful module: it holds a per-cell threshold map, resamples it at fabricate time, and broadcasts it to a reference tensor on request.

## Design decisions

- **`inst_shape` is an `__init__` argument, not a config field.** The per-instance shape is a property of the deployment instance, not of the device, so it is passed to the constructor and kept out of the frozen `SelectorConfig`. The config carries the nominal threshold and the mismatch sigma only.
- **Mismatch sigma stays in the config; the policy only gates it.** `vth_mismatch__V` is a direct `SelectorConfig` field, and `SelectorPolicy.vth_mismatch` is the boolean that decides whether the fabricate sample is applied; with the flag off the threshold map stays the uniform nominal.
- **`T__K` is accepted but unused.** The constructor takes the uniform device-construction context (`dtype`, `T__K`) for interface uniformity with the other devices; the selector model has no temperature dependence.

## Contracts & invariants

- **`_sample_fabricate_mismatch()` resamples the whole map.** It re-expands `_nominal_vth__V` to `self.inst_shape` and redraws every cell's `_vth__V` state rather than updating in place.
- **`sample_vth_like(reference)` is the read path.** It broadcasts the fabricated threshold to the reference tensor's shape. The reference tensor defines the target; the selector imposes no shape of its own.

## Performance & resources

State is one ordinary threshold tensor at `inst_shape` plus its 0-D nominal buffer. `sample_vth_like` broadcasts the fabricated tensor, without copying its underlying storage, to the reference shape.

## Gotchas

- **The selector publishes a threshold, not a conduction current.** Its only read surface, `sample_vth_like`, returns a threshold-voltage tensor. The model defines no switching law or current equation.

## Known limitations

- No test coverage. `Selector` has no dedicated test module, and no integration path exercises it either. A focused unit test (threshold sampling, mismatch statistics, broadcast shape) is a coverage gap.

---

- **Reference**: [selector](../../../reference/primitive/device/selector.md)
- **Implementation**: `neurox/primitive/device/selector.py`
- **Tests**: TODO — no dedicated selector tests yet
