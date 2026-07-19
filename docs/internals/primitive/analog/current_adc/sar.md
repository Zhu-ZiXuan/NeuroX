# Triple-margin current SAR ADC

How `SarCurrentAdc` (`current_adc/sar.py`) realises the triple-margin current SAR contract.

## Design decisions

- **One SA time-shared over the binary search and the columns.** The sub-comparisons are a method-internal Python loop, never separate profiled leaves, so `_convert_impl` emits exactly one dynamic-energy event and one latency event for the whole conversion. The fabricated `inst_shape` is the real shared sense-lane count, NOT one per column; the forward tensor still carries per-column data, so `inst_shape` sets only the PPA multiplicity and the serial-latency time-multiplex factor.
- **Static offsets sampled once, held across steps.** `_sample_fabricate_mismatch` draws the comparator and coupling input-referred offsets once at `inst_shape` (zero when their policy toggles are off) and holds them constant across every binary-search step. `convert` gathers them to the forward logical-column axis via `_col_to_lane` — the `n_col` columns partition into `n_lane = inst_shape[-1]` contiguous equal blocks, so column `c` maps to lane `(c * n_lane) // n_col`.
- **Offset added after the pre-gain.** The gathered offset is added to `margin_gain * (i_in - i_ref)` after the pre-gain, so the effective decision offset is `sigma / margin_gain` — the triple-margin benefit is expressed directly in the decision expression, not baked into the sampled sigma.
- **2-D `[mode][tap]` threshold bank, mode-selected per call.** `ref_levels__uA` holds one strictly increasing ladder of exactly `2**n_bits - 1` thresholds per operating mode (a flat in-code tuple canonicalizes to a single mode row in `__post_init__`); the buffer is `[mode_num, 2**n_bits - 1]`. `_convert_impl` bounds-checks `adc_operation_point.adc_mode` against `mode_num`, requires `adc_operation_point.adc_bits == n_bits`, and indexes the mode row before the binary search. The selection is quasi-static — switching modes emits no per-conversion energy, so a conversion on any mode row is event-identical to a single-mode ADC holding that ladder.
- **Control-based energy attribution.** The per-step energy bills only the `input_mirror_ratio`-scaled regeneration path the SA's sized mirror controls; the unity input and reference legs are owned by upstream blocks and not re-billed. The data-independent `e_fixed_per_op` is broadcast onto the per-column term each step, and the step energies are summed into one event.

## Contracts & invariants

- **MSB-first accumulation.** `_select_ref` reads the high resolved bits as an MSB-first prefix to index the selected mode row's mid-point threshold; `_set_bit` writes bit `n_bits - 1 - step`. The output is unsigned in `[0, 2**n_bits - 1]`.
- **`max_bits` = `config.n_bits`; `unsigned_range(b)` = `(0, 2**b - 1)`.** The magnitude resolution is fixed by the config; `unsigned_range` validates `1 <= b <= max_bits`.
- **Consumed operating point.** `_convert_impl` reads `adc_operation_point.adc_mode` as the ladder-row index (`ValueError` outside `[0, mode_num)`) and rejects any `adc_bits != n_bits` — this single-point quantizer digitizes at exactly its physical resolution.
- **Not-yet-modelled toggles.** `mirror_mismatch` / `mirror_mismatch_sigma_relative` and `replica_threshold_variation` are wired in the policy/config but left as a domain-author TODO in `_sample_fabricate_mismatch`.

---

- **Reference**: [triple-margin current SAR ADC](../../../../reference/primitive/analog/current_adc/sar.md)
- **Implementation**: `neurox/primitive/analog/current_adc/sar.py`
- **Tests**: `tests/primitive/analog/test_current_adc_modes.py`
