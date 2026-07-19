# Current ADC base

The current ADC family: the abstract `CurrentAdc` (`current_adc/base.py`) carries the registry, `from_config`, the concrete `convert` template method over the leaf-provided `_convert_impl` hook, and the abstract `max_bits` / `unsigned_range` surface. The ADC owns the digitize step alone; the sign is handled outside by the caller, and the code-to-scale rescale is the caller's too. The ADC self-holds no reference — the reference levels arrive on the concrete config.

## Design decisions

- **Family dispatch keyed on config type.** `CurrentAdc.from_config(...)` discriminates on `type(config)` through `RegistryMixin[type[CurrentAdcConfig], CurrentAdc]`; adding a concrete current ADC only adds a registration line and a config subclass, never touches `from_config`. `CurrentAdcConfig` is the family config base — it declares the shared static-PPA fields (`area_per_inst__um2`, `leakage_per_inst__uW`) once for every member ([base](../base.md)).
- **Empty marker `CurrentAdcPolicy`.** The base policy carries no switch; each concrete current ADC declares its own `*Policy(CurrentAdcPolicy)` with that topology's toggles.
- **Unsigned single-ended output, no zero shift.** The input is a non-negative magnitude and the output is an unsigned code in `[0, 2**adc_bits - 1]`; there is no offset-binary re-bias. The sign is the caller's concern — this keeps the ADC a pure magnitude quantizer and lets the sign path (e.g. a current subtractor upstream) own the sign bit.
- **Shared operating-point types.** `AdcOperationPoint` and `AdcCalibrationRecord` are the domain-neutral types from `neurox/primitive/analog/adc_common.py`, imported by both the voltage and current ADC families so neither depends on the other.
- **`convert` is a probe-emitting template method; `_convert_impl` is not `@abstractmethod`.** The base's concrete `convert` calls `self._convert_impl(...)` and then emits the call's input, `code`, and the `AdcOperationPoint` fields on the `adc.convert` probe channel (`AdcProber.ADC_CONVERT`; a no-op without an active prober, never touching the returned code). `_convert_impl` raises `NotImplementedError` instead of being declared abstract — a deliberate loosening so a capture-style subclass can override `convert` wholesale and stay instantiable without a conversion body.

## Contracts & invariants

- **Uniform construction.** `from_config` builds every registered impl through one call shape, so each concrete current ADC must accept the base's construction arguments unchanged — narrowing or reordering them breaks dispatch. The shared shape is why the base accepts `dtype` / `T__K` it never uses; the subclass captures them.

---

- **Reference**: [current ADC family](../../../../reference/primitive/analog/current_adc/family.md)
- **Implementation**: `neurox/primitive/analog/current_adc/base.py`
- **Tests**: TODO - name the guarding test
