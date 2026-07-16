# Voltage DAC base

The root of the voltage DAC family: `VoltageDac` provides the shared construction and registration surface every member inherits and fixes the abstract conversion surface each concrete impl must fill; the code-to-voltage transfer and its non-idealities belong to the member.

## Design decisions

- **Family dispatch keyed on config type.** `VoltageDac.from_config(...)` discriminates on `type(config)` through `RegistryMixin[type[VoltageDacConfig], VoltageDac]`; `VoltageDacConfig` is the family config base — it declares the shared static-PPA fields (`area_per_inst__um2`, `leakage_per_inst__uW`) once for every member ([base](../base.md)), is the type the registry keys on, and is extended by every concrete config. Adding a concrete voltage DAC adds a registration line and a config subclass, nothing in the base.
- **No family-specific runtime extras.** The voltage DAC has no per-call operating point analogous to the ADC's `(mode, bits)`; the `from_config` signature is the canonical one and the conversion takes only the code, off the memory- and compile-critical path.
- **Empty marker `VoltageDacPolicy`.** The base policy carries no switch; each concrete voltage DAC declares its own `*Policy(VoltageDacPolicy)` with that topology's toggles.

## Contracts & invariants

- **Uniform construction.** `from_config` builds every registered impl through one call shape, so each concrete voltage DAC must accept the base's construction arguments unchanged — narrowing or reordering them breaks dispatch. The shared shape is why the base accepts `dtype` / `T__K` it never uses; the subclass captures them.
- **Required subclass surface.** `convert(code)` (code → drive voltage) and the `code_max` property (the inclusive maximum valid code). Per-op latency is leaf-defined (fixed-latency impls carry `latency_per_op__ns` on their own config).
- **No default fabricate step.** `FabricateMixin` provides the auto-cascade `fabricate()` but declares `_sample_fabricate_mismatch` abstract (no default); each concrete voltage DAC implements its own — an explicit no-op when it introduces no static per-output mismatch, or a real sampling step otherwise.

---

- **Reference**: [voltage DAC family](../../../../reference/primitive/analog/voltage_dac/family.md)
- **Implementation**: `neurox/primitive/analog/voltage_dac/base.py`
- **Tests**: TODO - name the guarding test
