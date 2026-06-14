# `neurox/analog/switch_cap.py`

## Current role

`SwitchCap` is a bottom-plate-sampled capacitor bank that performs passive charge-share averaging across a fixed set of weighted caps. The bank's cap count and per-cap weights are bound at construction; the class is encoding-agnostic — the semantic meaning of those weights is not part of the SwitchCap contract.

## Config boundary

`SwitchCapConfig` carries the per-bank physical knobs only:

- `c_unit__fF` — unit-cell capacitance.
- `cap_mismatch_sigma_relative` — per-unit-cell Pelgrom mismatch sigma.
- `energy_per_sample_overhead__fJ`, `leakage_per_inst__uW`, `area_per_inst__um2`, `latency_per_op__ns` — PPA / spec.

The bank's cap-count and per-cap weights are **not** in the config — they are init-time arguments to `SwitchCap.__init__`. The config describes the cell, the constructor describes the deployment.

## Policy boundary

`SwitchCapPolicy` carries the per-run decisions:

- `cap_mismatch` — apply per-unit-cell Pelgrom mismatch at fabricate time.
- `sampling_thermal_noise` — apply per-bank kT/C settling noise at sample time.

## Lifecycle

- `__init__(*, config, policy, name, inst_shape, dtype, T__K, cap_weights)` — bind `inst_shape` (per-instance fabrication shape) and `cap_weights: tuple[float, ...]` (length `n_caps`, all positive); build the nominal cap-array buffer `nominal_c__fF = config.c_unit__fF · cap_weights`.
- `_sample_fabricate_mismatch()` — driven by `FabricateMixin.fabricate()`; clone-expands `nominal_c__fF` to `(*self._inst_shape, n_caps)` and applies Pelgrom-scaled static mismatch (gated by `self.policy.cap_mismatch`). Buffer is reassigned via attribute, not re-`register_buffer`.
- `sample_and_accumulate(v_in)` — per-call charge-share kernel: `Σ C_k V_k / Σ C_k`, with kT/C noise added when `self.policy.sampling_thermal_noise=True`.

`T__K` is required at `__init__` (not a config field) because it sets the kT/C noise sigma — an operating-state quantity, not a design parameter.

## Why `cap_weights` lives in `__init__`, not the fabrication body

The per-cap weights are a structural property of the bank: a SwitchCap instance committed to a given digit-encoding (binary, unit, …) does not change its weight template across re-fabricates. Placing both `cap_weights` and `inst_shape` in `__init__` matches the canonical leaf signature shared by NMOS, ADC, DAC, mux, driver, TIA — `_sample_fabricate_mismatch` becomes a pure shape-only resampling step.

`cap_weights` is a Python `tuple[float, ...]`; tensor construction happens once inside `SwitchCap.__init__`.

See also:

- `docs/dev/architecture/fabrication_lifecycle.md`
