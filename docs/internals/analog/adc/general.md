# General ADC

## Summary

`GeneralADC` (`adc/general.py`) is the boundary-bucketize behavioural ADC: a sorted threshold list plus two optional Gaussian noise stages.

## Design decisions

- **Zero code cached at `__init__`.** The bit width is fixed by the boundary list (single-mode), so the topology-specific midpoint `self._zero_code = n_codes // 2` is a static instance attribute committed once, not a per-call computation. This is the opposite of the multi-mode SAR variants, which must compute the zero code per call because `bits` is a runtime parameter.
- **No fabricated mismatch.** Both noise sources (sampling, comparator) are dynamic and applied inside `convert`; `_sample_fabricate_mismatch` is an explicit no-op and `fabricate()` only resolves the profiler-inst tally already locked at `__init__`.
- **Reference-free `convert`, with `v_refs__V` deled for protocol symmetry.** `GeneralADC.convert` takes the family `v_refs__V` keyword to match the ADC base signature but discards it (`del v_refs__V`), exactly as `signed_range` `del`s `adc_bits` — its floor-bucketize is reference-free, so no tap participates in the conversion.

## Contracts & invariants

- **Single-mode validation.** `convert` accepts only `adc_mode == 0` and `adc_bits` equal to the boundary-implied width; any other operating point is rejected.
- **Clamp before the zero shift.** Per the base clamp-before-shift contract ([base](base.md)), `convert` clamps to `[0, n_codes-1]` before subtracting `self._zero_code`; here the out-of-range push comes from `floor_bucketize`'s stochastic-rounding LSB jitter.
- **Monotone input transform.** `input_transform` is identity or `log2`; both are monotone, so the bucketize threshold comparison stays valid for either.

## Performance & resources

N/A - a single floor_bucketize per call, off the memory- and compile-critical path.

## Gotchas

- N/A.

## Known limitations

- `GeneralADC` is a placeholder behavioural ADC; its comparator model, boundary list, noise stages and input-unit handling are provisional and may change.

---

- **Reference**: [general](../../../reference/analog/adc/general.md)
- **Implementation**: `neurox/analog/adc/general.py`
- **Tests**: TODO - name the guarding test
