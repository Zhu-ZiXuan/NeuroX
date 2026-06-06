# `neurox/analog/adc/general.py`

## Current role

`GeneralADC` is the boundary-bucketize ADC: a sorted list of comparator thresholds plus three optional Gaussian-noise stages. It is the simplest behavioural ADC in the family.

## Config

`GeneralADCConfig(ADCConfig)` carries:

- `boundaries: list[float]` — sorted comparator thresholds in input units (uA for current-mode, V for voltage-mode).
- `drive_value: float` — clamp reference voltage. Ignored when the clamp voltage comes from a separate driver block.
- `input_transform: Literal["linear", "log2"]` — optional log-domain transform applied before bucketize.
- `sampling_noise__V`, `comparator_noise__V`, `drive_thermal__V` — three Gaussian noise sigmas.
- PPA / latency fields.

## Policy

`GeneralADCPolicy(ADCPolicy)`:

- `sampling_noise: bool` — apply `sampling_noise__V` at convert time.
- `comparator_noise: bool` — apply `comparator_noise__V` at convert time.
- `drive_thermal: bool` — apply `drive_thermal__V` at drive time.

## Convert pipeline

`convert(v_pos__V, v_neg__V, *, mode, bits)`:

1. Differential signal `v_pos − v_neg`.
2. Optional sampling noise (input-referred jitter).
3. Optional `log2` domain transform.
4. Optional comparator noise (per-threshold offset).
5. Optional uniform LSB-jitter for stochastic rounding when `self.training` is `True`.
6. `floor_bucketize` against `boundaries` → `int16` code.

The class is single-mode: `mode == 0` and `bits == n_bits_implied_by_boundaries` are the only legal runtime pair.

## Training-mode rounding

Stochastic-vs-deterministic rounding is driven exclusively by `self.training` (the standard `nn.Module` flag). Switch between `model.train()` and `model.eval()` to toggle behaviour; there is no per-instance override.

## Fabrication

`GeneralADC` carries no per-instance static mismatch state — all of its noise sources (sampling, comparator, drive thermal) are dynamic and applied inside `convert`. Its inherited `_sample_fabricate_mismatch` is therefore the default no-op; `fabricate()` calls just resolve to a profiler-inst tally already locked at `__init__`.

See also:

- `base.md`
- `docs/dev/modules/common/quant.md`
