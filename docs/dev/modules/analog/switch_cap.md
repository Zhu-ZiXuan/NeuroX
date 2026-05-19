# `neurox/analog/switch_cap.py`

## Current role

`SwitchCap` is a bottom-plate-sampled capacitor bank that performs passive charge-share averaging across a fixed set of weighted caps. The bank's cap count and per-cap weights are bound at construction; the class is encoding-agnostic — the semantic meaning of those weights is not part of the SwitchCap contract.

## Config boundary

`SwitchCapConfig` carries the per-bank physical knobs only:

- `c_unit__fF` — unit-cell capacitance.
- `cap_mismatch_sigma_relative` + `enable_cap_mismatch` — per-unit-cell Pelgrom mismatch and its toggle.
- `enable_sampling_thermal_noise` — toggle for the per-bank kT/C settling noise.
- `energy_per_sample_overhead__fJ`, `leakage_per_inst__uW`, `area_per_inst__um2`, `latency_per_op__ns` — PPA / spec.

The bank's cap-count and per-cap weights are **not** in the config — they are init-time arguments to `SwitchCap.__init__`. The config describes the cell, the constructor describes the deployment.

## Lifecycle

- `__init__(*, cfg, name, T__K, dtype, cap_weights)` — bind `cap_weights: tuple[float, ...]` (length `n_caps`, all positive) and build the nominal cap-array buffer `nominal_c__fF = cfg.c_unit__fF · cap_weights` directly. No separate unit-cap buffer is kept.
- `fabricate(shape)` — clone-expand `nominal_c__fF` to `(*shape, n_caps)` and apply Pelgrom-scaled static mismatch (gated by `cfg.enable_cap_mismatch`). The fabricated tensor lands at `(*shape, n_caps)`.
- `sample_and_accumulate(v_in)` — per-call charge-share kernel: `Σ C_k V_k / Σ C_k`, with kT/C noise added when `cfg.enable_sampling_thermal_noise=True`.

`T__K` is required at `__init__` (not a config field) because it sets the kT/C noise sigma — an operating-state quantity, not a design parameter.

## Why `cap_weights` lives in `__init__`, not `fabricate`

The per-cap weights are a structural property of the bank: a SwitchCap instance committed to a given digit-encoding (binary, unit, …) does not change its weight template across re-fabricates. Placing `cap_weights` in `__init__` lets `fabricate(shape)` stay shape-only — matching the canonical leaf-fabricate signature shared by NMOS, ADC, DAC, mux, driver, TIA, decoder — and lets the leaf own its own nominal-cap tensor, mirroring the `nominal_c__fF` pattern of `McsSarAdc`.

`cap_weights` is a Python `tuple[float, ...]`; tensor construction happens once inside `SwitchCap.__init__`.

See also:

- `docs/dev/architecture/fabrication_lifecycle.md`
