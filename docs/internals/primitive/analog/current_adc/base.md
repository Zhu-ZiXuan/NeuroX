# Single-ended current ADC base

The abstract `Iadc` carries config-keyed construction, the `convert` template method, and the shared current-quantization surface.

## Design decisions

- **Family dispatch keyed on config and policy types.** `Iadc.from_config(...)` resolves `(type(config), type(policy))`; adding a concrete current ADC registers its concrete pair and never touches `from_config`. A mismatched pair fails before leaf construction.
- **Empty marker `IadcPolicy`.** The base policy carries no switch; each concrete current ADC declares its own `*Policy(IadcPolicy)` with that topology's toggles.
- **Unsigned single-ended output, no zero shift.** The input is a non-negative magnitude and the output is an unsigned code in `[0, 2**bits - 1]`; there is no offset-binary re-bias.
- **Mode-free conversion.** `convert` receives the injected per-instance reference values `i_refs__uA` (taps on the last axis) and the resolution `bits`. No operating-mode object or rescale data enters the ADC interface: the owner names a mode to its reference source and the source returns the matching taps, so mode identity is consumed on the owner-to-reference edge and never reaches the owner-to-ADC edge.
- **Reference count is a circuit property, not an interface law.** How many reference values a conversion consumes is decided by the concrete converter's own circuit: a converter that takes one reference and multiplies it internally receives one, a converter wired to a bank of thresholds receives the bank. The base neither declares nor validates a tap count, so a ladder a leaf cannot use fails inside that leaf.
- **Bit width is ADC-internal.** The ladder states the converter's own wiring and does not follow the requested `bits`; a leaf realizes a lower resolution by running fewer decision cycles over it. For a deterministic leaf this makes the code at `bits` the code at `max_bits` right-shifted by `max_bits - bits` — the family's bit-width equivalence law.
- **`convert` is a record-emitting template method.** The base's concrete `convert` validates `bits` against the ADC's own capability (`_check_bits`), calls the abstract `self._convert_impl(...)`, and then, only `if IadcProber.active()`, builds and submits an `IadcRecord` (the input magnitude, the returned `code`, and the plain `bits` scalar) to `IadcProber` — demand-gated so a run nobody collects never touches the returned code or builds the record. The record covers the conversion event alone; a reference is a calibrated constant rather than a measured quantity, so it stays out of it.
- **Duration is declared here, with the executed bit count as its whole input.** `latency__ns(*, bits)` is abstract on `Iadc`: a conversion is the only thing this family spends time on, and how long one lasts follows from the resolution the call executes. The formula is the leaf's — a flat comparison window, a sum over search steps — so the base supplies no default and no `latency_per_op__ns` field.
- **`IadcProber` is co-located with its record in this module.** `IadcProber(RecorderBase[IadcRecord])` sits beside `IadcRecord` in `current_adc/base.py`, next to `Iadc`, the family's sole emitter.

## Contracts & invariants

- **Uniform construction.** `from_config` builds every registered impl through one call shape, so each concrete current ADC must accept the base's construction arguments unchanged — narrowing or reordering them breaks dispatch. The shared shape is why the base accepts `dtype` / `T__K` it never uses; the subclass captures them.
- **The base owns the `bits` contract and nothing else.** `bits` is base semantics because `unsigned_range` is declared here, so `_check_bits` requires `bits` in `[1, max_bits]` against the leaf's own declared capability and no leaf repeats that check; `unsigned_range` reuses it for the same bound. Everything else about the call — the reference count above all — is the concrete converter's, checked by that converter or by the consumer that wires it.

---

- **Reference**: [single-ended current ADC family](../../../../reference/primitive/analog/current_adc/family.md)
- **Implementation**: `neurox/primitive/analog/current_adc/base.py`
- **Tests**: `tests/primitive/analog/test_current_adc.py`
