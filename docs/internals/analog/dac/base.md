# DAC base

## Summary

The DAC family: the abstract `DAC` (`dac/base.py`) carrying the registry, `from_config`, and profiler registration only. Concrete impls live alongside ([general](general.md)). Spec: [reference/analog/dac/base](../../../reference/analog/dac/base.md).

## Design decisions

- **Family dispatch keyed on config type.** `DAC.from_config(...)` discriminates on `type(config)` through `RegistryMixin[type[DACConfig], DAC]`; `DACConfig` is an empty marker so a parent config has a base type to name as the polymorphic field. Adding a concrete DAC adds a registration line and a config subclass, nothing in the base.
- **No family-specific runtime extras.** The DAC has no per-call operating point analogous to the ADC's `(mode, bits)`; the `from_config` signature is the canonical one and the conversion takes only the code.
- **Empty marker `DACPolicy`.** As with the ADC family, the base policy carries no switch; concrete impls declare their own `*Policy(DACPolicy)`, and the composite stores the abstract field type.

## Contracts & invariants

- **Family init signature.** `__init__(*, config, policy, name, inst_shape, dtype, T__K)` - all six keyword-only and non-`None`. The base stores `self._inst_shape` and uses `name` for profiler registration; `config / policy / dtype / T__K` stay on the concrete subclass.
- **Required subclass surface.** `convert(code)` (code -> voltage) and `code_to_signal` (the code -> nominal-voltage LUT, read directly for a nominal operating-point voltage without the noisy convert). Static PPA comes from `CircuitBase`; per-op latency is leaf-defined (fixed-latency impls carry `latency_per_op__ns` on their own config).
- **No default fabricate step.** `FabricateMixin` provides the auto-cascade `fabricate()` but no default `_sample_fabricate_mismatch`; each concrete DAC declares its own — an explicit no-op when it introduces no static per-output mismatch (as `GeneralDAC` does), or a real sampling step otherwise.

## Performance & resources

N/A - a table lookup off the memory- and compile-critical path.

## Gotchas

- N/A.

## Known limitations

- N/A.

---

- **Reference**: [dac base](../../../reference/analog/dac/base.md)
- **Implementation**: `neurox/analog/dac/base.py`
- **Tests**: TODO - name the guarding test
- **Decisions**: N/A — no ADR governs this module.
