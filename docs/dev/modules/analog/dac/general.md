# `neurox/analog/dac/general.py`

## Current role

`GeneralDAC` is the simplest concrete DAC: a code → voltage LUT plus optional additive Gaussian drive-thermal noise.

## Config

`GeneralDACConfig(DACConfig)` carries:

- `code_to_signal: list[float]` — LUT entry per integer code.
- `drive_thermal__V: float` — additive Gaussian noise sigma on the output.
- `energy_per_op__fJ`, `latency_per_op__ns`, `leakage_per_inst__uW`, `area_per_inst__um2` — PPA / spec.

## Policy

`GeneralDACPolicy(DACPolicy)`:

- `drive_thermal: bool` — apply `drive_thermal__V` at convert time.

## Lifecycle

- `__init__(*, config, policy, name, inst_shape, dtype, T__K)` builds the `code_to_signal` LUT buffer, forwards `name` and `inst_shape` to the family base, and records the instance count for the profiler.
- `convert(code)` — index `code_to_signal`, apply `drive_thermal__V` gated by `self.policy.drive_thermal`, emit per-call dynamic energy through the profiler side channel.
- `_sample_fabricate_mismatch` is the inherited default no-op — `GeneralDAC` has no static per-instance state.

The DAC's per-element energy is fully captured by `energy_per_op__fJ`. Set it to `0` whenever the same switching energy is accounted at another stage to avoid double-counting.

See also:

- `base.md`
