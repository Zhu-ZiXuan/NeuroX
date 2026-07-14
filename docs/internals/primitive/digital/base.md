# Digital base

`DigitalBase` is the base for the integer-datapath family — a sized `ModuleBase[DigitalConfigT, DigitalPolicy]` + `ProfileMixin`. It carries the fabricate no-op and the static-PPA override once for the whole family; the two per-family roots `DigitalConfig` and `DigitalPolicy` sit beside it.

## Design decisions

- **Static PPA is placed once at the base.** The digital family is homogeneous — every leaf's area / leakage is the same `config.<field>_per_inst__* × inst_count` aggregate — so `DigitalBase` composes `ProfileMixin` and defines the `area__um2` / `leakage__uW` properties once, and each leaf inherits them. Placing the override at the base rather than per leaf is safe precisely because no digital leaf sizes itself differently.
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
