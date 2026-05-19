# `neurox/analog/switch_cap.py`

## Current role

`SwitchCap` is a bottom-plate-sampled capacitor bank that performs passive charge-share averaging across an arbitrary set of weighted caps. The class is encoding-agnostic — a `SwitchCap` is just a bank of caps with externally supplied weights; the semantic meaning of those weights is not part of the SwitchCap contract.

## Config boundary

`SwitchCapConfig` carries the per-bank physical knobs only:

- `c_unit__fF` — absolute cap scale.
- `cap_mismatch_sigma_relative` — per-unit-cell Pelgrom mismatch (`None` disables).
- `enable_thermal_noise` — toggle for the per-cap kT/C noise.
- `energy_per_sample_overhead__fJ`, `leakage_per_inst__uW`, `area_per_inst__um2`, `latency_per_op__ns` — PPA / spec.

The bank's number of caps and their relative weights are **not** in the config — they are runtime inputs to `fabricate(shape, cap_ratio)`. The config describes the cell, the fabricate call describes the deployment.

## Lifecycle

- `__init__(*, cfg, name, T__K, dtype)` — register the unit-cap nominal buffer.
- `fabricate(shape, cap_ratio)` — broadcast `c_unit__fF` to the requested bank shape, apply the per-cap weight template `cap_ratio`, then apply Pelgrom-scaled static mismatch. The fabricated tensor lands at `(*shape, n_caps)`.
- `sample_and_accumulate(v_in)` — per-call charge-share kernel: `Σ C_k V_k / Σ C_k`, with optional kT/C noise added when `cfg.enable_thermal_noise=True`.

`T__K` is required at `__init__` (not a config field) because it sets the kT/C noise sigma — it is an operating-state quantity, not a design parameter.

## Why the weights live in fabricate, not config

The same SwitchCap topology supports many per-cap weight templates (positional digit weight vectors, unit-weight banks, …). Keeping the weights as a `fabricate` argument rather than a config field decouples the leaf circuit from any one weight convention.

See also:

- `../../architecture/fabrication_lifecycle.md`
