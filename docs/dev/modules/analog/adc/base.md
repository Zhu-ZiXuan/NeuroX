# `neurox/analog/adc/base.py`

## Current role

`ADC` is the abstract base for the ADC family. It carries:

- the per-family `config_type → impl_class` registry (via `RegistryDispatchMixin[type[ADCConfig], ADC]`)
- the family-level `from_config(...)` classmethod
- profiler registration
- the abstract `convert(...)`, `latency_per_op__ns(*, bits)`, `area_per_inst__um2`, `leakage_per_inst__uW` contracts every concrete ADC must implement.

`ADCConfig` is the empty family-base marker used by the `RegistryDispatchMixin` dispatch surface (every concrete ADC config subclasses it). `ADCMode` is the small `(n_bits, n_states, max_signal)` dataclass used by the multi-mode subclasses' calibration LUTs.

## Family-wide init signature

Every concrete ADC impl exposes the same explicit signature:

```
__init__(self, *, cfg, name, inst_shape, dtype, T__K)
```

`inst_shape` is the per-instance fabrication shape, committed at construction. The base stores `self._inst_shape` and accepts/discards `cfg / dtype / T__K` so the dispatcher type-checks; concrete subclasses store them on `self`. `ADC` inherits `FabricateMixin`: subclasses override `_sample_fabricate_mismatch` to refresh static state; the cascading `fabricate()` is auto-implemented by the mixin. Stochastic-vs-deterministic rounding is governed by `self.training` at `convert` time — there is no constructor-time override flag.

## Runtime multi-mode

`convert(v_pos__V, v_neg__V, *, mode, bits)` and `latency_per_op__ns(*, bits)` take their operating point as **per-call** keyword arguments. Single-mode subclasses honour the contract by validating `mode == 0` and `bits == max_bits`; multi-mode SAR variants accept any pair inside their configured envelope.

## Floor semantics

ADC boundaries are placed at code edges `B_c = c · LSB`. Stochastic rounding adds `uniform(0, LSB)` jitter before the floor and is unbiased. This matches the `floor_bucketize` kernel in [`docs/dev/modules/common/quant.md`](docs/dev/modules/common/quant.md).

## What ADC does not own

- Clamp voltage and current-to-voltage conversion (those live in a separate `ClampDriver` block such as a `TIA`).
- Analog-domain rescale factors that convert ADC codes back to an ideal-integer scale.
- Column multiplexing (`AnalogMux`).

See also:

- `general.md`, `mcs_sar.md`, `sar_mono.md`
- `docs/dev/modules/analog/tia/README.md`
- `docs/dev/modules/common/registry_dispatch.md`
