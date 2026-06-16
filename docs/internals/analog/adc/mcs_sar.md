# MCS SAR ADC — Implementation

## Summary

`McsSarAdc` (`adc/mcs_sar.py`) is the production multi-mode SAR ADC: differential merged-capacitor switching with fabricated cap-mismatch and comparator-offset state. Spec: [reference/analog/adc/mcs_sar](../../../reference/analog/adc/mcs_sar.md).

## Design decisions

- **Zero code computed per call, not cached.** Under multi-mode operation `bits` is a per-call runtime parameter, so the zero code `2**(bits-1)` is computed inline in `convert` after the unsigned clamp; there is no per-instance cache (the opposite of [general](general.md), whose width is fixed).
- **One instance covers the full envelope.** Every per-cell term scales with the runtime `mode` (reference-voltage entry) and `bits` (active SAR depth), so a single fabricated instance serves all operating points; `bits < max_bits` simply leaves the smaller caps idle. No per-mode instance is constructed.
- **No `latency_per_op__ns` field.** Per-op latency is `(adc_operation_point.adc_bits + 1) * clk_period__ns`, derived in `convert` from the runtime op point and emitted through the profiler latency side channel alongside the per-conversion energy - the SAR latency genuinely depends on the runtime depth, so a static field would be wrong.
- **Comparator-noise sigma temperature-scaled at `__init__`.** The configured comparator-thermal-noise sigma is anchored at 300 K and scaled by `sqrt(T/300)` at construction, because `T__K` is an operating-state quantity bound at construction, not a re-fabricate-time resample.
- **Independent positive / negative CDAC legs.** Cap-mismatch is fabricated independently per leg, matching the physical two-array differential topology; a shared mismatch draw would understate the differential error.

## Contracts & invariants

- **Fabricated state at `_inst_shape`.** `_sample_fabricate_mismatch` (driven by `FabricateMixin.fabricate()`) samples the per-cap Pelgrom mismatch (both legs) and the static comparator offset; the nominal cap / comparator-offset buffers are seeded at `__init__`.
- **Clamp precedes the per-call zero shift** (family contract): after the SAR loop the unsigned code is clamped to `[0, 2**bits - 1]` before subtracting `2**(bits-1)`.
- **Multi-mode per-call kwargs.** `mode` indexes `v_refs__V` (strictly decreasing, entry 0 the calibration anchor); `bits <= max_bits` sets the active depth.

## Performance & resources

The SAR loop is `bits` sequential decision cycles. To keep the unrolled compiled graph short, `convert` hoists the input-independent per-bit constants out of the loop: the used-cap slices feed precomputed tables (`v_p_step_table__V`, `v_n_step_table__V`, the per-bit switch-energy tables `e_step_p_table__fJ` / `e_step_n_table__fJ`, and `c_diff_step_table__fF`), since these depend only on the fabricated caps and `v_ref` / `v_cm`, not on the runtime input. The loop body keeps only the input-dependent work — indexing the step tables by `k`, the `torch.where` top-plate updates, the comparator strobe, and the code shift — and the energy / `c_diff` side-channels are then computed vectorised from the final code bits. `convert` runs on the macro compiled path (readout sits inside `vec_mat_mul`), so keeping the unrolled graph short keeps `bits` sequential decision cycles from bloating it. TODO - add measured per-conversion latency / memory once benchmarked.

## Gotchas

- **`bits < max_bits` leaves caps idle, it does not rescale them.** The loop engages only the top `bits - 1` caps; the smaller caps contribute no switching energy. Treating a reduced-depth conversion as a full-depth one with scaled caps would mis-account energy.

## Known limitations

- **No measured compile / benchmark numbers recorded here yet** - the SAR-loop compile constraints are stated but the latency / memory profile is a TODO.

---

- **Reference**: [mcs_sar](../../../reference/analog/adc/mcs_sar.md)
- **Implementation**: `neurox/analog/adc/mcs_sar.py`, `neurox/analog/adc/_multimode.py`
- **Tests**: TODO - name the guarding test
- **Decisions**: N/A — no ADR governs this module.
