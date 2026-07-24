# MCS SAR voltage ADC

## Design decisions

- **Per-resolution constant tables dodge a dynamo `1 << SymInt` miscompile.** `bits` is a per-call runtime parameter, so the unsigned clamp bound `2**bits - 1` and the zero offset `2**(bits-1)` vary per conversion. Evaluating `1 << bits` inside `_convert_impl` would emit a `1 << <SymInt>` op that dynamo's lshift lowering currently mishandles, so `__init__` precomputes both as plain-`int` tuples (`_unsigned_max_table`, `_zero_offset_table`) that `_convert_impl` and the `unsigned_range` / `zero_offset` accessors index by the runtime `bits`.
- **One instance covers the full resolution envelope.** The per-call `bits` value selects the active SAR depth; the selected `v_ref__V` tensor is supplied directly.
- **No `latency_per_op__ns` field.** Per-op latency `(bits + 1) * clk_period__ns` is derived in `_convert_impl` from the runtime op point and emitted through the profiler latency side channel — the SAR latency depends on the runtime depth, so a static config field would be wrong.
- **Comparator-noise sigma temperature-scaled at `__init__`.** `_comparator_noise_sigma__V` is scaled once at construction, not resampled at fabricate, because `T__K` is bound at construction; Reference gives the `sqrt(T)` law.
- **Independent positive / negative CDAC legs.** Cap-mismatch is drawn independently for the two legs (`_c_p__fF`, `_c_n__fF`); a shared draw would understate the differential error.

## Contracts & invariants

- **Fabricated state at `inst_shape`.** `_sample_fabricate_mismatch` (driven by `FabricateMixin.fabricate()`) samples the per-cap Pelgrom mismatch (both legs) and the static comparator offset; the nominal cap / comparator-offset buffers are seeded at `__init__`.
- **Raw offset-binary code returned, zero point exposed not folded.** `_convert_impl` returns the raw offset-binary SAR code in `[0, 2**bits - 1]`; `_zero_offset_table[bits]` is exposed through `zero_offset(bits)` and never subtracted inside the ADC. No standalone unsigned clamp runs — the SAR loop leaves `code` in range by construction, and `apply_lsb_jitter` re-clamps after its `+1` overflow.
- **Per-call operating point.** `v_ref__V` broadcasts through the tensor-valued `v_cm`, step-table, and energy math; `bits <= max_bits` sets the active depth. No mode index enters the ADC.

## Performance & resources

The SAR loop is `bits` sequential decision cycles. To keep an unrolled compiled graph short, `_convert_impl` hoists the input-independent per-bit constants out of the loop: the used-cap slices feed precomputed tables (`v_p_step_table__V`, `v_n_step_table__V`, the per-bit switch-energy tables `e_step_p_table__fJ` / `e_step_n_table__fJ`, and `c_diff_step_table__fF`), since these depend only on the fabricated caps and `v_ref` / `v_cm`, not on the runtime input. The loop body keeps only input-dependent indexing, top-plate updates, comparator strobe, and code shift; energy and `c_diff` are computed vectorised from the final code bits.

## Gotchas

- **`bits < max_bits` leaves caps idle, it does not rescale them.** The loop engages only the top `bits - 1` caps; the smaller caps contribute no switching energy. Treating a reduced-depth conversion as a full-depth one with scaled caps would mis-account energy.
- **Training-mode LSB jitter makes `convert` non-deterministic.** After the SAR loop the code passes through `apply_lsb_jitter(..., enabled=self.training)` — a Bernoulli(0.5) +0/+1 LSB stochastic-rounding fallback gated by `nn.Module.training`, not a policy flag. Left at the default `training=True`, the output is per-call random even under an all-off policy; call `.eval()` for deterministic, chunk-bit-exact codes.

---

- **Reference**: [mcs_sar](../../../../reference/primitive/analog/diff_voltage_adc/mcs_sar.md)
- **Implementation**: `neurox/primitive/analog/diff_voltage_adc/mcs_sar.py`
- **Tests**: `tests/primitive/analog/test_adc_family.py`
