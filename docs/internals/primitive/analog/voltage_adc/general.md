# General voltage ADC

## Design decisions

- **Zero code cached at `__init__`.** The bit width is fixed by the boundary list, so the topology-specific midpoint `self._zero_code = n_codes // 2` is committed once. It is exposed through `zero_code` and `zero_offset(bits)` but never folded into the raw code.
- **No fabricated mismatch.** Both noise sources (sampling, comparator) are dynamic and applied inside `_convert_impl`; `_sample_fabricate_mismatch` is an explicit no-op and `fabricate()` only resolves the profiler-inst tally already locked at `__init__`.
- **Reference-free `_convert_impl`.** `GeneralDifferentialVoltageAdc._convert_impl` accepts the family `v_ref__V` keyword but discards it; its fixed boundary list fully determines the conversion.
- **Training-mode stochastic rounding.** `_convert_impl` forwards `training=self.training` to `floor_bucketize`, so under `train()` a uniform `U[0, LSB)` dither is added before the floor (the realized code is stochastic) and under `eval()` the bucketize is the plain deterministic floor. `LSB` is `self._lsb_estimate` — the mean threshold spacing (`(boundaries[1:] - boundaries[:-1]).mean()`, or the lone value for a single-threshold list) committed at `__init__`. This realizes the family `self.training` rounding gate ([base](base.md)) through the boundary spacing.

## Contracts & invariants

- **Fixed-resolution validation.** `_convert_impl` accepts only `bits` equal to the boundary-implied width.
- **Clamp to the raw range.** Per the base raw-code contract ([base](base.md)), `_convert_impl` clamps to `[0, n_codes-1]` and returns the raw bucket unshifted; here the out-of-range push comes from `floor_bucketize`'s stochastic-rounding LSB jitter.
- **Monotone input transform.** `input_transform` is identity or `log2`; both are monotone, so the bucketize threshold comparison stays valid for either.

## Performance & resources

A single `floor_bucketize` per call, off the memory- and compile-critical path.

- **Per-convert energy / latency self-log.** After the bucketize, `_convert_impl` attributes its own runtime energy and latency through two profiler side-channel emits (both no-ops outside a profiler, neither touching the returned code): `_log_dynamic_energy` receives `torch.full_like(code, energy_per_op__fJ, dtype=torch.float32)` — one `energy_per_op__fJ` charge per output-code element, so the profiler's `.sum()` totals `energy_per_op__fJ * code.numel()`; `_log_latency` receives the scalar `latency_per_op__ns * serial_op_count`. Both are static config fields (fixed-per-op leaf).
- **Serial-op count.** `serial_op_count = max(1, code.numel() // max(self.inst_count, 1))` divides the total output elements by the fabrication multiplicity `inst_count = prod(inst_shape)` — fabricated instances convert in parallel, the remaining leading (batch) elements pass serially through the shared converter. GeneralDifferentialVoltageAdc's `code` carries no parallel trailing dim beyond `inst_shape`, so this position-invariant numel ratio is the exact serial count; the outer `max(1, ...)` floors it at one op and `max(self.inst_count, 1)` guards a zero divisor.

## Known limitations

- The model is boundary-based and contains no transistor-level comparator,
  reference-generation, or settling model.

---

- **Reference**: [general](../../../../reference/primitive/analog/voltage_adc/general.md)
- **Implementation**: `neurox/primitive/analog/voltage_adc/general.py`
- **Tests**: `tests/primitive/analog/test_adc_family.py`, `tests/primitive/analog/test_voltage_adc_probe.py`
