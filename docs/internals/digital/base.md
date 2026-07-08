# Digital base

`DigitalCircuit` is the thin base for the integer-datapath family: it owns no state and its only family-wide addition is the fabricate no-op.

## Design decisions

- **Only a fabricate no-op.** A digital block holds no analog device state, so it has no static manufacturing variation to resample; `DigitalCircuit` states this once for the family by implementing `_sample_fabricate_mismatch` as an explicit no-op, and each leaf inherits it rather than re-declaring the hook.

## Contracts & invariants

- **`fabricate()` is a family-wide pass-through.** The base no-op resamples nothing and no digital leaf adds fabricated mismatch, so `fabricate()` introduces no per-instance variation on any leaf.

---

- **Reference**: [digital](../../reference/digital/README.md)
- **Implementation**: `neurox/digital/base.py`
- **Tests**: TODO - no dedicated digital test module yet
