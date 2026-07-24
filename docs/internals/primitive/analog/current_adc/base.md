# Current ADC base

The current ADC family: the abstract `SingleEndedCurrentAdc` (`current_adc/base.py`) carries the registry, `from_config`, the concrete `convert` template method over the leaf-provided `_convert_impl` hook, and the abstract `max_bits` / `unsigned_range` surface. The ADC owns the digitize step alone; the sign is handled outside by the caller, and the code-to-scale rescale is the caller's too. The ADC self-holds no reference and knows nothing of operating modes — a per-instance ladder `i_refs__uA` of shape `[*R, n_ref]` with `n_ref = 2**bits - 1` as the last axis (taps ascending), already reduced to the mode's row by the caller's reference block, and the resolution `bits` arrive per `convert` call directly. The `[*R]` leading broadcasts right-aligned against `i_in__uA`, so a per-element reference ladder is quantized without collapse.

## Design decisions

- **Family dispatch keyed on config type.** `SingleEndedCurrentAdc.from_config(...)` discriminates on `type(config)` through `RegistryMixin[type[SingleEndedCurrentAdcConfig], SingleEndedCurrentAdc]`; adding a concrete current ADC only adds a registration line and a config subclass, never touches `from_config`. `SingleEndedCurrentAdcConfig` is the family config base — it declares the shared static-PPA fields (`area_per_inst__um2`, `leakage_per_inst__uW`) once for every member ([base](../base.md)).
- **Empty marker `SingleEndedCurrentAdcPolicy`.** The base policy carries no switch; each concrete current ADC declares its own `*Policy(SingleEndedCurrentAdcPolicy)` with that topology's toggles.
- **Unsigned single-ended output, no zero shift.** The input is a non-negative magnitude and the output is an unsigned code in `[0, 2**bits - 1]`; there is no offset-binary re-bias. The sign is the caller's concern — this keeps the ADC a pure magnitude quantizer and lets the sign path (e.g. a current subtractor upstream) own the sign bit.
- **Mode-free convert; rescale lives upstream.** The single-ended ADC takes `bits` directly and never sees an operating mode — the caller (the composing macro) selects the mode's ladder row before the call. Neither ADC family takes an operating-point object: both the current and voltage families receive a pre-selected reference plus `bits`, with mode selection performed upstream by the owner/macro. `AdcCalibrationRecord` (the `(mode, bits) -> rescale_factor` row from `adc_common`) is a macro/unit-layer rescale structure, not an ADC input.
- **`convert` is a probe-emitting template method; `_convert_impl` is not `@abstractmethod`.** The base's concrete `convert` calls `self._convert_impl(...)` and then, only `if SingleEndedCurrentAdcProber.active()`, builds and submits a `SingleEndedCurrentAdcObservation` (the input magnitude, the returned `code`, and the plain `bits` scalar) to `SingleEndedCurrentAdcProber` — demand-gated so an unsubscribed run never touches the returned code or builds the payload. `_convert_impl` raises `NotImplementedError` instead of being declared abstract — a deliberate loosening so a capture-style subclass can override `convert` wholesale and stay instantiable without a conversion body.
- **`SingleEndedCurrentAdcProber` is co-located with its payload in this module.** `SingleEndedCurrentAdcProber(Prober[SingleEndedCurrentAdcObservation])` sits beside `SingleEndedCurrentAdcObservation` in `current_adc/base.py`, next to `SingleEndedCurrentAdc`, the observation link's sole emitter.

## Contracts & invariants

- **Uniform construction.** `from_config` builds every registered impl through one call shape, so each concrete current ADC must accept the base's construction arguments unchanged — narrowing or reordering them breaks dispatch. The shared shape is why the base accepts `dtype` / `T__K` it never uses; the subclass captures them.
- **`record_latency` passthrough.** `__init__` and `from_config` accept `record_latency: bool = True` and forward it down to `ModuleBase`, which stores `self.record_latency`. A leaf gates its `_log_latency` emission on this flag; dynamic energy is always logged. A composing macro passes `record_latency=False` when it owns the sole latency event and the ADC's own step latency must not double-count.

---

- **Reference**: [current ADC family](../../../../reference/primitive/analog/current_adc/family.md)
- **Implementation**: `neurox/primitive/analog/current_adc/base.py`
- **Tests**: `tests/primitive/analog/test_current_adc.py`
