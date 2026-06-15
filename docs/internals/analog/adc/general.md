# general ADC — Implementation

## Summary

`GeneralADC` (`adc/general.py`) is the boundary-bucketize behavioural ADC: a sorted threshold list plus three optional Gaussian noise stages. Spec: [reference/analog/adc/general](../../../reference/analog/adc/general.md).

## Design decisions

- **Zero code cached at `__init__`.** The bit width is fixed by the boundary list (single-mode), so the topology-specific midpoint `self._zero_code = n_codes // 2` is a static instance attribute committed once, not a per-call computation. This is the opposite of the multi-mode SAR variants, which must compute the zero code per call because `bits` is a runtime parameter.
- **No fabricated mismatch.** All three noise sources (sampling, comparator, drive-thermal) are dynamic and applied inside `convert`; `_sample_fabricate_mismatch` is the inherited no-op and `fabricate()` only resolves the profiler-inst tally already locked at `__init__`.

## Contracts & invariants

- **Single-mode validation.** `convert` accepts only `adc_mode == 0` and `adc_bits` equal to the boundary-implied width; any other operating point is rejected.
- **Clamp precedes the zero shift** (family contract): the optional stochastic LSB jitter can push the floor_bucketize index out of `[0, n_codes-1]`, so the clamp runs before subtracting `self._zero_code`.
- **Monotone input transform.** `input_transform` is identity or `log2`; both are monotone, so the bucketize order is preserved and the threshold comparison stays valid.

## Performance & resources

N/A - a single floor_bucketize per call, off the memory- and compile-critical path.

## Gotchas

- **`drive_value` is ignored when a separate driver clamps.** The config field exists for standalone use; in the readout chain the clamp voltage comes from the boundary driver block, so the field is inert there - do not read it as the operating clamp.

## Known limitations

- N/A.

---

- **Reference**: [general](../../../reference/analog/adc/general.md)
- **Implementation**: `neurox/analog/adc/general.py`
- **Tests**: TODO - name the guarding test
- **Decisions**: N/A — no ADR governs this module.
