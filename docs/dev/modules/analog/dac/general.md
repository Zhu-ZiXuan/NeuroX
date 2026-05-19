# `neurox/analog/dac/general.py`

## Current role

`GeneralDAC` is the simplest concrete DAC: a code → voltage LUT plus optional additive Gaussian drive-thermal noise.

## Config

`GeneralDACConfig(DACConfig)` carries:

- `code_to_signal: list[float]` — LUT entry per integer code.
- `drive_thermal: float | None` — Gaussian noise sigma (`None` disables).
- `energy_per_op__fJ`, `latency_per_op__ns`, `leakage_per_inst__uW`, `area_per_inst__um2` — PPA / spec.

## Lifecycle

- `__init__` registers the LUT as a non-persistent buffer (`code_to_signal`) and forwards `name` to the family base for profiler registration. `T__K` is stored but unused — kept for the uniform analog construction signature.
- `convert(code)` — index `code_to_signal`, apply `drive_thermal` if set, emit per-call dynamic energy through the profiler side channel.
- `fabricate(shape)` — inherited no-op; the LUT has no shape-derived state.

The DAC's per-element energy is fully captured by `energy_per_op__fJ`. Set it to `0` whenever the same switching energy is accounted at another stage to avoid double-counting.

See also:

- `base.md`
