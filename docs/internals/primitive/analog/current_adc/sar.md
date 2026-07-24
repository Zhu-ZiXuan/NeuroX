# Triple-margin current SAR ADC

How `SarSingleEndedCurrentAdc` (`current_adc/sar.py`) realises the triple-margin current SAR contract.

## Design decisions

- **One SA time-shared over the binary search and the columns.** The sub-comparisons are a method-internal Python loop, never separate profiled leaves, so `_convert_impl` emits exactly one dynamic-energy event (and, when `record_latency` is set, one latency event) for the whole conversion. The fabricated `inst_shape` is the real shared sense-lane count, NOT one per column; the forward tensor still carries per-column data, so `inst_shape` sets only the PPA multiplicity and the serial-latency time-multiplex factor.
- **Static offsets sampled once, held across steps.** `_sample_fabricate_mismatch` draws the comparator and coupling input-referred offsets once at `inst_shape` (zero when their policy toggles are off) and holds them constant across every binary-search step. `convert` gathers them to the forward logical-column axis via `_col_to_lane` — the `n_col` columns partition into `n_lane = inst_shape[-1]` contiguous equal blocks, so column `c` maps to lane `(c * n_lane) // n_col`.
- **Offset added after the pre-gain.** The gathered offset is added to `margin_gain * (i_in - i_ref)` after the pre-gain, so the effective decision offset is `sigma / margin_gain` — the triple-margin benefit is expressed directly in the decision expression, not baked into the sampled sigma.
- **Per-call per-instance reference ladder + `bits`.** The ladder is not a config field: `convert` / `_convert_impl` take `i_refs__uA` and `bits` per call (the caller's reference block is the single source and has already reduced the bank to the operating mode's row). `i_refs__uA` has shape `[*R, n_ref]` with `n_ref == 2**bits - 1` taps ascending along the last axis; the `[*R]` leading broadcasts right-aligned against `i_in__uA`, so each forward element resolves against its own per-instance ladder. `_convert_impl` requires `bits` in `[1, config.bits]` and `n_taps == 2**bits - 1`, broadcasts the ladder to `(*i_in.shape, n_taps)` once, and runs the binary search with a per-element `torch.gather` along the tap axis. Mode is invisible to the ADC, so switching modes emits no per-conversion energy.
- **B-form per-step energy.** Each step adds the fixed switching constant `e_fixed_per_op`, the current-domain conduction `v_rail__V * (i_in + i_ref_step) * t_conduct_per_step__ns[step]` (`V * uA * ns = fJ`), and the `_input_dynamic_energy__fJ` hook (base returns a zero broadcast — the escape hatch for a structural subclass, kept off the parameterized path). All-zero `t_conduct_per_step__ns` collapses to the pure `bits * e_fixed_per_op` model. The step energies are summed into one event; the unity input and reference legs sourced upstream are billed there, not re-billed here.

## Contracts & invariants

- **MSB-first accumulation.** `_select_ref(ref_b, code, step, bits)` reads the high resolved bits of `code` as an MSB-first prefix (`prefix = code >> (bits - step)`), forms the mid-point tap index, and gathers it from the broadcast ladder `ref_b` along the tap axis; `_set_bit` writes bit `bits - 1 - step`. The output is unsigned in `[0, 2**bits - 1]`.
- **`max_bits` = `config.bits`; `unsigned_range(bits)` = `(0, 2**bits - 1)`.** `config.bits` is the physical (maximum) resolution; a call requests any `bits` in `[1, config.bits]`, and `unsigned_range` validates that bound.
- **Per-call resolution windows.** The conduction (`t_conduct_per_step__ns`) and latency (`step_latency__ns`) lists must be at least `config.bits` long; a call at `bits` sums only their first `bits` entries, so a shared shipped list covers several resolutions without overbilling.
- **Latency emission is construction-gated.** `record_latency` (constructor keyword, default `True`) gates only the latency event: `_convert_impl` always computes and emits the summed dynamic-energy event, but builds the latency tensor and calls `_log_latency` only when `record_latency` is set. A host that owns conversion timing (e.g. a macro folding the ADC window into its own cycle accounting) constructs the ADC with `record_latency=False`, so it contributes energy but no latency.
- **Not-yet-modelled toggles.** `mirror_mismatch` / `mirror_mismatch_sigma_relative` and `replica_threshold_variation` are wired in the policy/config but left as a domain-author TODO in `_sample_fabricate_mismatch`.

---

- **Reference**: [triple-margin current SAR ADC](../../../../reference/primitive/analog/current_adc/sar.md)
- **Implementation**: `neurox/primitive/analog/current_adc/sar.py`
- **Tests**: `tests/primitive/analog/test_current_adc.py`
