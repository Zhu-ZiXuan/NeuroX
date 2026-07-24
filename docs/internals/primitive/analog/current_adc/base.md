# Current ADC base

The abstract `SingleEndedCurrentAdc` carries config-keyed construction, the `convert` template method, and the shared current-quantization surface.

## Design decisions

- **Family dispatch keyed on config type.** `SingleEndedCurrentAdc.from_config(...)` discriminates on `type(config)` through `RegistryMixin[type[SingleEndedCurrentAdcConfig], SingleEndedCurrentAdc]`; adding a concrete current ADC only adds a registration line and a config subclass, never touches `from_config`. `SingleEndedCurrentAdcConfig` is the family config base — it declares the shared static-PPA fields (`area_per_inst__um2`, `leakage_per_inst__uW`) once for every member ([base](../base.md)).
- **Empty marker `SingleEndedCurrentAdcPolicy`.** The base policy carries no switch; each concrete current ADC declares its own `*Policy(SingleEndedCurrentAdcPolicy)` with that topology's toggles.
- **Unsigned single-ended output, no zero shift.** The input is a non-negative magnitude and the output is an unsigned code in `[0, 2**bits - 1]`; there is no offset-binary re-bias.
- **Mode-free conversion.** `convert` receives an ascending per-instance ladder `i_refs__uA` with trailing length `2**bits - 1` and the resolution `bits`. No operating-mode object or rescale data enters the ADC interface.
- **`convert` is a probe-emitting template method; `_convert_impl` is not `@abstractmethod`.** The base's concrete `convert` calls `self._convert_impl(...)` and then, only `if SingleEndedCurrentAdcProber.active()`, builds and submits a `SingleEndedCurrentAdcObservation` (the input magnitude, the returned `code`, and the plain `bits` scalar) to `SingleEndedCurrentAdcProber` — demand-gated so an unsubscribed run never touches the returned code or builds the payload. `_convert_impl` raises `NotImplementedError` instead of being declared abstract — a deliberate loosening so a capture-style subclass can override `convert` wholesale and stay instantiable without a conversion body.
- **`SingleEndedCurrentAdcProber` is co-located with its payload in this module.** `SingleEndedCurrentAdcProber(Prober[SingleEndedCurrentAdcObservation])` sits beside `SingleEndedCurrentAdcObservation` in `current_adc/base.py`, next to `SingleEndedCurrentAdc`, the observation link's sole emitter.

## Contracts & invariants

- **Uniform construction.** `from_config` builds every registered impl through one call shape, so each concrete current ADC must accept the base's construction arguments unchanged — narrowing or reordering them breaks dispatch. The shared shape is why the base accepts `dtype` / `T__K` it never uses; the subclass captures them.
- **`enable_latency_record` passthrough.** `__init__` and `from_config` accept `enable_latency_record: bool = True` and forward it to `ModuleBase`. A leaf gates only its latency emission on this flag; dynamic energy remains enabled.

---

- **Reference**: [current ADC family](../../../../reference/primitive/analog/current_adc/family.md)
- **Implementation**: `neurox/primitive/analog/current_adc/base.py`
- **Tests**: `tests/primitive/analog/test_current_adc.py`
