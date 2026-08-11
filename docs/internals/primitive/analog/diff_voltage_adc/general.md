# General voltage ADC

## Design decisions

- **Code count is config, thresholds are data.** `code_num` is the only transfer field on the config: the comparator count `code_num - 1` is topology and is known at construction, while the threshold voltages are injected per call. This keeps `max_bits`, `unsigned_range` and `zero_code` init-determined constants — the midpoint `self._zero_code = code_num // 2` is committed once, exposed through `zero_code` and `zero_offset(bits)`, and never folded into the raw code.
- **No fabricated mismatch.** Both noise sources (sampling, comparator) are dynamic and applied inside `_convert_impl`; `_sample_fabricate_mismatch` is an explicit no-op and `fabricate()` only resolves the profiler-inst tally already locked at `__init__`.
- **Injected thresholds.** `_convert_impl` bucketizes against the family `v_refs__V` keyword; the ADC self-holds no reference. The bank is the whole flat comparator set, so the leaf states its own circuit fact in `_validate_runtime_args`: a 1-D ladder of exactly `code_num - 1` taps. `torch.bucketize` genuinely requires 1-D boundaries, so the shape does not speak for itself.
- **Duration is the flat comparison window.** `latency__ns(*, bits)` returns `latency_per_op__ns` unchanged — the bank fires every ladder tap at once, so the converter owns no time axis and the executed resolution does not lengthen the window. The config field is therefore the whole timing model.
- **Training-mode stochastic rounding.** `_convert_impl` forwards `training=self.training` to `floor_bucketize`, so under `train()` a uniform `U[0, LSB)` dither is added before the floor (the realized code is stochastic) and under `eval()` the bucketize is the plain deterministic floor. `LSB` is `_lsb__V(v_refs__V)` — the mean tap spacing of the injected ladder, or the lone tap when the bank holds one — computed per call, because the thresholds are not known at construction. This realizes the family `self.training` rounding gate ([base](base.md)) through the threshold spacing.

## Contracts & invariants

- **Fixed-resolution validation.** `_convert_impl` accepts only `bits` equal to the code-count-implied width.
- **Clamp to the raw range.** Per the base raw-code contract ([base](base.md)), `_convert_impl` clamps to `[0, code_num-1]` and returns the raw bucket unshifted; here the out-of-range push comes from `floor_bucketize`'s stochastic-rounding LSB jitter.

## Performance & resources

A single `floor_bucketize` per call, off the memory- and compile-critical path.

- **Per-convert energy self-log.** After the bucketize, `_convert_impl` attributes its own runtime energy through one profiler side-channel emit without changing the returned code. The energy is a flat per-conversion lump, so the emission expands a 0-dim constant onto the code's layout: the expanded view holds no storage, nothing is materialized, and the energy dtype comes from the constant rather than from the integer code.
- **No instance axis reaches the payload.** `inst_shape` never enters this forward path — it only sizes the static PPA via `inst_count` — and the comparator ladder is one shared 1-D bank with no per-instance buffer touching the signal, so the emission has no instance axis to fold.

## Known limitations

- The model is threshold-based and contains no transistor-level comparator,
  reference-generation, or settling model.

---

- **Reference**: [general](../../../../reference/primitive/analog/diff_voltage_adc/general.md)
- **Implementation**: `neurox/primitive/analog/diff_voltage_adc/general.py`
- **Tests**: `tests/primitive/analog/test_adc_family.py`, `tests/primitive/analog/test_diff_voltage_adc_probe.py`
