# `neurox/analog/adc/mcs_sar.py`

## Current role

`McsSarAdc` is the V_cm-based (Merged Capacitor Switching) differential SAR ADC. The model carries explicit cap-mismatch and comparator-noise modelling and supports calibrated multi-mode operation.

## Topology

Differential MCS with bottom-plate sampling:

1. **Sample** — bottom plates of two independent CDAC arrays track the positive / negative inputs; top plates float. On release the bottom plates snap to `V_cm`, leaving each top plate at `V_ref − V_in`.
2. **MSB decision** — free comparison of the two top plates with the configured static offset and per-cycle thermal noise.
3. **SAR loop** — each cycle picks one cap on each leg and switches it from `V_cm` to `V_ref` or to GND according to the previous bit, moving the top plates by `± V_cm · C_k / C_total`.

The differential topology resolves the MSB through free comparison; no MSB cap is needed.

## Config

`McsSarAdcConfig(ADCConfig)`:

- `max_bits` — physical CDAC depth. The active array has `max_bits - 1` binary-weighted caps + a dummy unit cap.
- `v_refs__V: tuple[float, ...]` — strictly-decreasing reference voltages. `v_refs__V[0]` is the maximum (calibration anchor).
- `clk_period__ns`, `c_unit__fF` — design parameters.
- `cap_mismatch_sigma_relative`, `comparator_offset_sigma__V`, `comparator_thermal_noise_sigma__V` — non-ideality magnitudes.
- `e_bootstrap__fJ`, `e_constant_per_bit__fJ` — energy overhead.
- `area_per_inst__um2`, `leakage_per_inst__uW` — static PPA fields inherited from `CircuitConfig`. **No `latency_per_op__ns` field**: per-op latency is `(adc_operation_point.adc_bits + 1) × clk_period__ns`, computed inside `convert(...)` from the runtime op point.

## Policy

`McsSarAdcPolicy(ADCPolicy)`:

- `cap_mismatch` — apply per-cap Pelgrom mismatch at fabricate time.
- `comparator_offset` — apply static comparator offset at fabricate time.
- `comparator_thermal_noise` — apply per-SAR-cycle comparator thermal noise.
- `sampling_thermal_noise` — apply kT/C sampling noise on the held top plates.

## Lifecycle

- `__init__(..., inst_shape, ...)` — temperature-scale the comparator-noise sigma (σ ∝ √T anchored at 300 K) and seed nominal cap / comparator-offset buffers. The per-instance shape is committed here.
- `_sample_fabricate_mismatch()` (driven by `fabricate()` via `FabricateMixin`) — sample static state per instance: per-cap Pelgrom mismatch (independent positive / negative CDAC legs) and comparator threshold offset, at `self._inst_shape`.
- `convert(v_pos__V, v_neg__V, *, mode, bits)` — sample with optional kT/C noise, do the SAR loop with per-cycle comparator noise, optionally add LSB jitter, clamp to `[0, 2**bits - 1]`, then subtract the per-call zero code `2**(bits - 1)` to return a signed code in `[-2**(bits-1), 2**(bits-1) - 1]`. The zero code is **not** cached on the instance because `bits` is a per-call runtime parameter under multi-mode operation.

## Multi-mode support

`(mode, bits)` are per-call kwargs:

- `mode` selects the `v_refs__V` entry to use.
- `bits` (≤ `max_bits`) controls the active SAR depth. When `bits < max_bits` the loop engages only the top `bits − 1` caps; smaller caps stay idle.

## Energy model

Per-conversion energy is a sum of:

- one-shot sampling energy (charging bottom plates from `V_cm` to `V_in`)
- per-cycle MCS formula `½ · V_ref² · C_k · (1 − C_k / C_total)`
- reset / cap-mismatch dump energy at the end of conversion
- `e_bootstrap__fJ` + `bits · e_constant_per_bit__fJ` lump-sum overhead

Every per-cell term scales naturally with the runtime `mode` and `bits`, so a single SAR instance covers the full multi-mode operating envelope.

See also:

- `base.md`
