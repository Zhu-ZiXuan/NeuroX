# Digital base

`DigitalBase` is the base for the integer-datapath family. It carries the fabricate no-op once for the whole family; the two per-family roots `DigitalConfig` and `DigitalPolicy` sit beside it.

## Design decisions

- **Every digital leaf reports its own static PPA.** Each leaf keeps `is_profile_target` at its default and binds its per-instance area and leakage in `__init__` for the profiler's static walk. The family has no separate aggregate reporting node.
- **`DigitalConfig` carries the PPA fields; `DigitalPolicy` is empty.** `DigitalConfig` declares and validates `area_per_inst__um2` / `leakage_per_inst__uW` as the family config base every leaf config extends. `DigitalPolicy` is an empty marker — an integer-exact block carries no non-ideality to toggle — but the pairing stays uniform, so each leaf's `__init__` still takes a required `policy: DigitalPolicy`.
- **Only a fabricate no-op.** A digital block holds no analog device state, so it has no static manufacturing variation to resample; `DigitalBase` states this once for the family by implementing `_sample_fabricate_mismatch` as an explicit no-op, and each leaf inherits it rather than re-declaring the hook.
- **One energy quantum per adder evaluation, counted on the operands.** The family bills the same physical thing everywhere: the number of adder cells a call fires. A reducing leaf therefore bills its pre-reduction operand and an element-wise leaf its result, because a fold spends one evaluation per leg it consumes — an adder tree fires one cell per leg, a serial register one accumulate per arrival — while an element-wise call fires one per result element. The reduced extent stays inside the billed element count; billing the result of a fold instead would make an N-way reduction cost the same as a 2-way one and hide the very axis the leaf exists to consume.
- **Not a registry family.** The digital blocks share no polymorphic dispatch surface, so each stays a leaf circuit instantiated by concrete class name and is kept out of `RegistryMixin`. No runtime type selection is part of the family contract.

## Contracts & invariants

- **Leaf construction takes `config` / `policy`.** A leaf's `__init__` forwards `config=..., policy=..., name=..., inst_shape=...` to `DigitalBase` (i.e. `ModuleBase`); the leaf config extends `DigitalConfig` (the inherited PPA fields first, then its own arithmetic / energy / latency fields) and `policy` is the empty `DigitalPolicy`.
- **`fabricate()` is a family-wide pass-through.** The base no-op resamples nothing and no digital leaf adds fabricated mismatch, so `fabricate()` introduces no per-instance variation on any leaf.
- **Dynamic energy is one profiler emission per call** (`_record_dynamic_energy`): the per-op quantum is flat per billed element — the family rule above fixes which tensor's layout that is — so the emission expands a 0-dim constant onto that layout, a view with no storage, and nothing is materialized. The module reduces nothing itself; the profiler owns that aggregation.
- **A digital leaf's `inst_shape` never reaches the datapath.** It sizes the static PPA totals only; no buffer shaped by it touches the billed tensor, so the energy tensor has no instance axis to fold.
- **A digital leaf owns no time axis, so it reports no duration.** The elementwise or reduced axes it works over belong to its caller, so no leaf carries a `latency__ns` method: `latency_per_op__ns` stays a config field — a circuit property of the block — and the caller that inserted the axes reads that constant and multiplies it by the round count only the caller knows.

---

- **Reference**: N/A — each digital leaf has its own physical specification
- **Implementation**: `neurox/primitive/digital/base.py`
- **Tests**: TODO - no dedicated digital test module yet
