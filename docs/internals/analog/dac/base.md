# DAC base

## Summary

The DAC family: the abstract `DAC` (`dac/base.py`) carrying the registry, `from_config`, profiler registration, and the abstract `convert` / `code_max` surface. Concrete impls live alongside.

## Design decisions

- **Family dispatch keyed on config type.** `DAC.from_config(...)` discriminates on `type(config)` through `RegistryMixin[type[DACConfig], DAC]`; `DACConfig` is an empty marker so a parent config has a base type to name as the polymorphic field. Adding a concrete DAC adds a registration line and a config subclass, nothing in the base.
- **No family-specific runtime extras.** The DAC has no per-call operating point analogous to the ADC's `(mode, bits)`; the `from_config` signature is the canonical one and the conversion takes only the code, off the memory- and compile-critical path.
- **Empty marker `DACPolicy`.** As with the ADC family, the base policy carries no switch; concrete impls declare their own `*Policy(DACPolicy)`, and the composite stores the abstract field type.

## Contracts & invariants

- **Family init signature.** `__init__(*, config, policy, name, inst_shape, dtype, T__K)` - all six keyword-only and non-`None`. The base forwards `config / name / inst_shape` to `CircuitBase` (storing `self.config` and `self._inst_shape` and registering `name` with the profiler) and discards `policy / dtype / T__K` via `del`; the concrete subclass re-stores `policy / dtype / T__K`.
- **Required subclass surface.** `convert(code)` (code → drive voltage) and the `code_max` property (the maximum valid code; valid codes lie in `[0, code_max]`). Static PPA comes from `CircuitBase`; per-op latency is leaf-defined (fixed-latency impls carry `latency_per_op__ns` on their own config).
- **No default fabricate step.** `FabricateMixin` provides the auto-cascade `fabricate()` but declares `_sample_fabricate_mismatch` abstract (no default); each concrete DAC implements its own — an explicit no-op when it introduces no static per-output mismatch, or a real sampling step otherwise.

---

- **Reference**: [dac base](../../../reference/analog/dac/family.md)
- **Implementation**: `neurox/analog/dac/base.py`
- **Tests**: TODO - name the guarding test
