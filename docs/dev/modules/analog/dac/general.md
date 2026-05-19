# `neurox/analog/dac/general.py`

## Current role

`GeneralDAC` is the simplest concrete DAC: a code → voltage LUT plus optional additive Gaussian drive-thermal noise.

## Config

`GeneralDACConfig(DACConfig)` carries:

- `code_to_signal: list[float]` — LUT entry per integer code.
- `drive_thermal__V: float` + `enable_drive_thermal: bool` — additive Gaussian noise sigma on the output and its toggle.
- `energy_per_op__fJ`, `latency_per_op__ns`, `leakage_per_inst__uW`, `area_per_inst__um2` — PPA / spec.

## Lifecycle

- `__init__` builds the `code_to_signal` LUT buffer and forwards `name` to the family base.
- `convert(code)` — index `code_to_signal`, apply `drive_thermal__V` gated by `enable_drive_thermal`, emit per-call dynamic energy through the profiler side channel.
- `fabricate(shape)` — no shape-derived state; the body only records the instance count.

The DAC's per-element energy is fully captured by `energy_per_op__fJ`. Set it to `0` whenever the same switching energy is accounted at another stage to avoid double-counting.

See also:

- `base.md`
