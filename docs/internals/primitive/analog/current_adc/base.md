# Single-ended current ADC base

The abstract `Iadc` carries config-keyed construction, the `convert` template method, and the shared current-quantization surface.

## Design decisions

- **Family dispatch keyed on config and policy types.** `Iadc.from_config(...)` resolves `(type(config), type(policy))`; adding a concrete current ADC registers its concrete pair and never touches `from_config`. A mismatched pair fails before leaf construction.
- **Empty marker `IadcPolicy`.** The base policy carries no switch; each concrete current ADC declares its own `*Policy(IadcPolicy)` with that topology's toggles.
- **Unsigned single-ended output, no zero shift.** The input is a non-negative magnitude and the output is an unsigned code in `[0, 2**bits - 1]`; there is no offset-binary re-bias.
- **Mode-free conversion.** `convert` receives the injected per-instance reference values `i_refs__uA` (taps on the last axis) and the resolution `bits`. No operating-mode object or rescale data enters the ADC interface: the owner names a mode to its reference source and the source returns the matching taps, so mode identity is consumed on the owner-to-reference edge and never reaches the owner-to-ADC edge.
- **Reference count is a circuit property, not an interface law.** How many reference values a conversion consumes is decided by the concrete converter's own circuit: a converter that takes one reference and multiplies it internally receives one, a converter wired to a bank of thresholds receives the bank. The base neither declares nor validates a tap count, so a ladder a leaf cannot use fails inside that leaf.
- **Bit width is ADC-internal.** The ladder states the converter's own wiring and does not follow the requested `bits`; a leaf realizes a lower resolution by running fewer decision cycles over it. For a deterministic leaf this makes the code at `bits` the code at `max_bits` right-shifted by `max_bits - bits` — the family's bit-width equivalence law.
- **`convert` is a probe-emitting template method.** The base's concrete `convert` validates `bits` against the ADC's own capability (`_check_bits`), calls the abstract `self._convert_impl(...)`, and then, only `if IadcProber.active()`, builds and submits a `IadcObservation` (the input magnitude, the returned `code`, and the plain `bits` scalar) to `IadcProber` — demand-gated so an unsubscribed run never touches the returned code or builds the payload. The observation records the conversion event alone; a reference is a calibrated constant rather than a measured quantity, so it stays out of the payload.
- **`IadcProber` is co-located with its payload in this module.** `IadcProber(Prober[IadcObservation])` sits beside `IadcObservation` in `current_adc/base.py`, next to `Iadc`, the observation link's sole emitter.

## Contracts & invariants

- **Uniform construction.** `from_config` builds every registered impl through one call shape, so each concrete current ADC must accept the base's construction arguments unchanged — narrowing or reordering them breaks dispatch. The shared shape is why the base accepts `dtype` / `T__K` it never uses; the subclass captures them.
- **The base owns the `bits` contract and nothing else.** `bits` is base semantics because `unsigned_range` is declared here, so `_check_bits` requires `bits` in `[1, max_bits]` against the leaf's own declared capability and no leaf repeats that check; `unsigned_range` reuses it for the same bound. Everything else about the call — the reference count above all — is the concrete converter's, checked by that converter or by the consumer that wires it.
- **`enable_latency_record` passthrough.** `__init__` and `from_config` accept `enable_latency_record: bool = True` and forward it to `ModuleBase`. A leaf gates only its latency emission on this flag; dynamic energy remains enabled.

---

- **Reference**: [single-ended current ADC family](../../../../reference/primitive/analog/current_adc/family.md)
- **Implementation**: `neurox/primitive/analog/current_adc/base.py`
- **Tests**: `tests/primitive/analog/test_current_adc.py`
