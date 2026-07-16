# MCS SAR voltage ADC

## Design decisions

- **Per-resolution constant tables dodge a dynamo `1 << SymInt` miscompile.** `bits` is a per-call runtime parameter, so the unsigned clamp bound `2**bits - 1` and the zero offset `2**(bits-1)` vary per conversion. Evaluating `1 << bits` inside `convert` would emit a `1 << <SymInt>` op that dynamo's lshift lowering currently mishandles, so `__init__` precomputes both as plain-`int` tuples (`_unsigned_max_table`, `_zero_offset_table`) that `convert` indexes by the runtime `bits`.
- **One instance covers the full envelope.** The runtime `mode` and `bits` select an injected reference tap and the active SAR depth, so a single fabricated instance serves every operating point; no per-mode instance is constructed.
- **The `mode` bound is checked against the injected tensor.** Because the reference ladder is injected per call, `_validate_runtime_args` keeps only the config-knowable `bits` bound; the `0 <= mode < v_refs__V.shape[-1]` check lives in `convert`, where the injected tensor is in hand.
- **No `latency_per_op__ns` field.** Per-op latency `(bits + 1) * clk_period__ns` is derived in `convert` from the runtime op point and emitted through the profiler latency side channel — the SAR latency depends on the runtime depth, so a static config field would be wrong.
- **Comparator-noise sigma temperature-scaled at `__init__`.** `comparator_noise_sigma__V` is scaled once at construction, not resampled at fabricate, because `T__K` is bound at construction; Reference gives the `sqrt(T)` law.
- **Independent positive / negative CDAC legs.** Cap-mismatch is drawn independently for the two legs (`c_p__fF`, `c_n__fF`); a shared draw would understate the differential error.

## Contracts & invariants

- **Fabricated state at `_inst_shape`.** `_sample_fabricate_mismatch` (driven by `FabricateMixin.fabricate()`) samples the per-cap Pelgrom mismatch (both legs) and the static comparator offset; the nominal cap / comparator-offset buffers are seeded at `__init__`.
- **Offset-binary code re-biased by table subtraction.** `convert` returns `code - _zero_offset_table[bits]`, re-biasing the offset-binary SAR code to two's complement per the family signed-code-range convention; the bias is subtracted (rather than flipping the MSB) to preserve the int32 storage of negatives. No standalone unsigned clamp runs — the SAR loop leaves `code` in `[0, 2**bits - 1]` by construction, and `apply_lsb_jitter` re-clamps after its `+1` overflow.
- **Multi-mode per-call kwargs.** `mode` indexes the injected `v_refs__V` (shape `(*inst, num_refs)`): `v_ref__V = v_refs__V[..., mode]` is a `Tensor` tap that broadcasts through the tensor-valued `v_cm` / step-table / energy math, and `bits <= max_bits` sets the active depth.

## Performance & resources

The SAR loop is `bits` sequential decision cycles. To keep the unrolled compiled graph short, `convert` hoists the input-independent per-bit constants out of the loop: the used-cap slices feed precomputed tables (`v_p_step_table__V`, `v_n_step_table__V`, the per-bit switch-energy tables `e_step_p_table__fJ` / `e_step_n_table__fJ`, and `c_diff_step_table__fF`), since these depend only on the fabricated caps and `v_ref` / `v_cm`, not on the runtime input. The loop body keeps only the input-dependent work — indexing the step tables by `k`, the `torch.where` top-plate updates, the comparator strobe, and the code shift — and the energy / `c_diff` side-channels are then computed vectorised from the final code bits. `convert` runs eager by default but stays compile-friendly: a caller that compiles the model then unrolls only this shortened loop. TODO - add measured per-conversion latency / memory once benchmarked.

## Gotchas

- **`bits < max_bits` leaves caps idle, it does not rescale them.** The loop engages only the top `bits - 1` caps; the smaller caps contribute no switching energy. Treating a reduced-depth conversion as a full-depth one with scaled caps would mis-account energy.
- **Training-mode LSB jitter makes `convert` non-deterministic.** After the SAR loop the code passes through `apply_lsb_jitter(..., enabled=self.training)` — a Bernoulli(0.5) +0/+1 LSB stochastic-rounding fallback gated by `nn.Module.training`, not a policy flag. Left at the default `training=True`, the output is per-call random even under an all-off policy; call `.eval()` for deterministic, chunk-bit-exact codes.

---

- **Reference**: [mcs_sar](../../../../reference/primitive/analog/voltage_adc/mcs_sar.md)
- **Implementation**: `neurox/primitive/analog/voltage_adc/mcs_sar.py`
- **Tests**: TODO - name the guarding test
