# Digital base

## Summary

The digital family: the thin `DigitalCircuit` base (`neurox/digital/base.py`) shared by every integer-datapath leaf. It owns no state and exists only to declare the family-wide fabricate hook. The concrete leaves live alongside.

## Design decisions

- **Thin base, only a fabricate no-op.** `DigitalCircuit` owns no state and exists only to declare the fabricate hook as one explicit family-wide no-op, since integer-exact logic has no static manufacturing variation to resample. There is nothing else on the base for a leaf to inherit.

## Contracts & invariants

- **`fabricate()` is a family-wide pass-through.** No digital leaf holds fabricated mismatch, so the shared `_sample_fabricate_mismatch` no-op on the base is correct for every one and each leaf's `fabricate()` introduces no per-instance variation.

---

- **Reference**: [digital](../../reference/digital/README.md)
- **Implementation**: `neurox/digital/base.py`
- **Tests**: TODO - no dedicated digital test module yet
