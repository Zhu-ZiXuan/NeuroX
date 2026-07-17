# Digital base

`DigitalBase` is the base for the integer-datapath family. It carries the fabricate no-op once for the whole family; the two per-family roots `DigitalConfig` and `DigitalPolicy` sit beside it.

## Design decisions

- **Every digital leaf reports its own static PPA.** A digital block's silicon is its own — none of it rolls up to an owning circuit — so the family leaves `is_profile_target` at its default and each leaf binds its per-instance area / leakage in `__init__` for the profiler's static walk to collect. The family is homogeneous — no leaf sizes itself differently.
- **`DigitalConfig` carries the PPA fields; `DigitalPolicy` is empty.** `DigitalConfig` declares `area_per_inst__um2` / `leakage_per_inst__uW` (plus `validate_ppa`) as the family config base every leaf config extends. `DigitalPolicy` is an empty marker — an integer-exact block carries no non-ideality to toggle — but the pairing stays uniform, so each leaf's `__init__` still takes a required `policy: DigitalPolicy`.
- **Only a fabricate no-op.** A digital block holds no analog device state, so it has no static manufacturing variation to resample; `DigitalBase` states this once for the family by implementing `_sample_fabricate_mismatch` as an explicit no-op, and each leaf inherits it rather than re-declaring the hook.
- **Not a registry family.** The digital blocks share no polymorphic dispatch surface, so each stays a leaf circuit instantiated by concrete class name and is kept out of `RegistryMixin`. Adding family dispatch would impose a config-type discriminator with no caller that needs to select among the blocks at runtime.

## Contracts & invariants

- **Leaf construction takes `config` / `policy`.** A leaf's `__init__` forwards `config=..., policy=..., name=..., inst_shape=...` to `DigitalBase` (i.e. `ModuleBase`); the leaf config extends `DigitalConfig` (the inherited PPA fields first, then its own arithmetic / energy / latency fields) and `policy` is the empty `DigitalPolicy`.
- **`fabricate()` is a family-wide pass-through.** The base no-op resamples nothing and no digital leaf adds fabricated mismatch, so `fabricate()` introduces no per-instance variation on any leaf.
- **Energy and latency are two independent profiler emissions** (`_log_dynamic_energy` then `_log_latency`): the energy tensor is per-output-element while the latency is a single scalar scaled by the serial-op count. One event does not carry both.
- **`inst_count` is guarded against zero** in the serial-op divisor (`max(inst_count, 1)`), so a leaf's serial-op math must not assume a positive instance count.

---

- **Reference**: [digital](../../../reference/primitive/digital/README.md)
- **Implementation**: `neurox/primitive/digital/base.py`
- **Tests**: TODO - no dedicated digital test module yet
