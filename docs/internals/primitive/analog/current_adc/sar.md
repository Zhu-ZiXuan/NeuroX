# Triple-margin current SAR ADC

How `SarIadc` (`current_adc/sar.py`) realises the triple-margin current SAR contract.

## Design decisions

- **One SA time-shared over the binary search and the columns.** The sub-comparisons are a method-internal Python loop, never separate profiled leaves, so `_convert_impl` emits exactly one dynamic-energy event (and, when `enable_latency_record` is set, one latency event) for the whole conversion. The fabricated `inst_shape` is the real shared sense-lane count, NOT one per column; the forward tensor still carries per-column data, so `inst_shape` sets only the PPA multiplicity and the serial-latency time-multiplex factor.
- **Static offsets sampled once, held across steps.** `_sample_fabricate_mismatch` draws the comparator and coupling input-referred offsets once at `inst_shape` (zero when their policy toggles are off) and holds them constant across every binary-search step. `convert` gathers them to the forward logical-column axis via `_col_to_lane` — the `n_col` columns partition into `n_lane = inst_shape[-1]` contiguous equal blocks, so column `c` maps to lane `(c * n_lane) // n_col`.
- **Offset added after the pre-gain.** The gathered offset is added to `margin_gain * (i_in - i_ref)` after the pre-gain, so the effective decision offset is `sigma / margin_gain` — the triple-margin benefit is expressed directly in the decision expression, not baked into the sampled sigma.
- **Per-call per-instance reference ladder + `bits`.** The ladder is not a config field: `convert` / `_convert_impl` take `i_refs__uA` and `bits` per call. `i_refs__uA` has shape `[*R, n_ref]` with `n_ref == 2**max_bits - 1` taps ascending along the last axis; the `[*R]` leading broadcasts right-aligned against `i_in__uA`, so each forward element resolves against its own per-instance ladder. The base validates `bits` and the tap count; `_convert_impl` broadcasts the ladder to `(*i_in.shape, n_taps)` once and runs the search with a per-element `torch.gather` along the tap axis.
- **Truncated binary search realizes a lowered resolution.** The search tree is always the `max_bits` one over the full ladder — the first compare sits at tap `2**(max_bits - 1) - 1` regardless of `bits` — and a `bits`-bit call simply stops after `bits` levels. Those levels are the max-bits code's leading bits, so the accumulator carries the code in max-bits numbering and `_convert_impl` returns it right-shifted by `max_bits - bits`. Nothing subsets the taps, so the search is identical to the max-bits one wherever the two overlap.
- **B-form per-step energy.** Each step adds the fixed switching constant `e_fixed_per_op`, the current-domain conduction `v_rail__V * (i_in + i_ref_step) * t_conduct_per_step__ns[step]` (`V * uA * ns = fJ`), and the `_compute_input_dynamic_energy__fJ` hook. All-zero `t_conduct_per_step__ns` collapses to the pure `bits * e_fixed_per_op` model. The step energies are summed into one event; source-generation energy for the input and reference currents is excluded.

## Contracts & invariants

- **MSB-first accumulation in max-bits numbering.** `_select_ref(ref_b, code, step, max_bits)` reads the high resolved bits of `code` as an MSB-first prefix (`prefix = code >> (max_bits - step)`), forms the mid-point tap index, and gathers it from the broadcast ladder `ref_b` along the tap axis; `_set_bit` writes bit `max_bits - 1 - step`. The returned output is unsigned in `[0, 2**bits - 1]`.
- **`max_bits` = `config.bits`; `unsigned_range(bits)` = `(0, 2**bits - 1)`.** `config.bits` is the physical (maximum) resolution; a call requests any `bits` in `[1, config.bits]`, and `unsigned_range` validates that bound.
- **Energy and latency count the EXECUTED steps.** The conduction (`t_conduct_per_step__ns`) and latency (`step_latency__ns`) lists must be at least `config.bits` long; a call at `bits` runs — and bills — only their first `bits` entries, so a shared shipped list covers several resolutions without overbilling.
- **Latency emission is construction-gated.** `enable_latency_record` (constructor keyword, default `True`) gates only the latency event: `_convert_impl` always computes and emits the summed dynamic-energy event, but builds the latency tensor and calls `_record_latency` only when `enable_latency_record` is set.

---

- **Reference**: [triple-margin current SAR ADC](../../../../reference/primitive/analog/current_adc/sar.md)
- **Implementation**: `neurox/primitive/analog/current_adc/sar.py`
- **Tests**: `tests/primitive/analog/test_current_adc.py`
