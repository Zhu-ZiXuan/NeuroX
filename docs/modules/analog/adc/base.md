# `neurox/analog/adc/base.py`

## Current role

`ADC` is the abstract base for the ADC family. It carries:

- the per-family `config_type → impl_class` registry (via `RegistryMixin[type[ADCConfig], ADC]`)
- the family-level `from_config(...)` classmethod
- profiler registration
- the abstract `convert(...)`, `mode_num`, `max_bits`, `signed_range(adc_bits)` contracts every concrete ADC must implement; per-instance static PPA (`area_per_inst__um2` / `leakage_per_inst__uW`) is provided by `CircuitBase`.

`ADCConfig` is the empty family-base marker used by the `RegistryMixin` dispatch surface (every concrete ADC config subclasses it). `ADCMode` is the small `(n_bits, n_states, max_signal)` dataclass used by the multi-mode subclasses' calibration LUTs. `AdcOperationPoint` is the frozen `(adc_mode, adc_bits)` runtime selection threaded into `convert(...)`; `AdcCalibrationRecord` is one row of the `(adc_mode, adc_bits) → rescale_factor` calibration table.

`ADCPolicy` is the empty marker base policy for the family. Concrete ADC impls declare their own `*Policy(ADCPolicy)` (e.g. `GeneralADCPolicy`, `SarAdcMonoPolicy`, `McsSarAdcPolicy`) carrying that topology's switches; the composite that holds an ADC stores the abstract `ADCPolicy` field type and the caller passes the concrete impl.

## Family-wide init signature

Every concrete ADC impl exposes the same explicit signature:

```
__init__(self, *, config, policy, name, inst_shape, dtype, T__K)
```

`inst_shape` is the per-instance fabrication shape, committed at construction. The base stores `self._inst_shape` and accepts/discards `config / policy / dtype / T__K` so the dispatcher type-checks; concrete subclasses store them on `self`. `ADC` inherits `FabricateMixin`: subclasses override `_sample_fabricate_mismatch` to refresh static state; the cascading `fabricate()` is auto-implemented by the mixin. Stochastic-vs-deterministic rounding is governed by `self.training` at `convert` time — there is no constructor-time override flag.

## Runtime multi-mode

`convert(v_pos__V, v_neg__V, *, adc_operation_point)` takes its operating point as a **per-call** keyword `AdcOperationPoint`. Single-mode subclasses honour the contract by validating `adc_mode == 0` and `adc_bits == max_bits`; multi-mode SAR variants accept any pair inside their configured envelope. `mode_num` exposes the number of supported operating points; `max_bits` exposes the maximum bit width.

## Per-op latency

Per-op latency is leaf-defined and emitted inside each concrete `convert(...)` body via `_log_latency(latency_tensor)` — there is no `latency_per_op__ns` contract on `ADC` / `ADCConfig`. Fixed-latency impls (`GeneralADC`) declare `latency_per_op__ns: float` on their own `*Config` and read `self.config.latency_per_op__ns × serial_op_count`. Parametric impls (`McsSarAdc`) do not carry the field at all; they derive `(adc_operation_point.adc_bits + 1) × self.config.clk_period__ns` and feed that into `_log_latency` alongside the conversion energy emitted by `_log_dynamic_energy`.

## Signed-code output convention

`convert(...)` returns **signed** integer codes in `[-2**(adc_bits-1), 2**(adc_bits-1) - 1]`. This is a family-level contract on the abstract base; how each concrete ADC converts from its native internal representation to the signed output is its own implementation detail.

The signed convention is required by the consumer model `M_ideal ≈ code · rescale_factor` (folded into the macro-side rescale; see [`docs/modules/macro/xbar/base.md`](../../macro/xbar/base.md) and `xbar/ideal.py`): a strictly positive `rescale_factor` mapping a signed code to a signed `M_ideal` is well-defined only when the ADC code carries the sign of the analog input directly. The xbar's `bl_adc` is differential and its v_diff is genuinely two-sided, so this convention aligns the physical ADC's output with `IdealXbar.vec_mat_mul` (which already produced signed codes) and with the calibrate tool's positivity invariant on `rescale_factor`.

How current concrete ADCs implement it:

- **`GeneralADC`**: bit width is fixed (boundary-implied), so the topology-specific zero code `n_codes // 2` is committed at construction as `self._zero_code`; `convert` clamps the raw bucket index to `[0, n_codes - 1]` and returns `code - self._zero_code`.
- **`McsSarAdc`**: bit width is per-call (`adc_operation_point.adc_bits`), so the zero code `2**(bits - 1)` is computed inline in `convert` after the unsigned clamp; no per-instance cache.

The "zero code" lives on each ADC as part of its topology knowledge rather than being centralised in a shared helper — different ADC families could in principle place their zero point differently (asymmetric boundaries, single-ended designs, …) and the base class deliberately makes no commitment.

Caller responsibility: each concrete `convert()` must clamp its raw unsigned output to its legal bucket range **before** the zero shift — stochastic-rounding jitter (e.g. from `floor_bucketize` or SAR LSB jitter) can push values outside the legal range and the subtraction would otherwise produce out-of-range signed codes.

## Floor semantics

ADC boundaries are placed at code edges `B_c = c · LSB`. Stochastic rounding adds `uniform(0, LSB)` jitter before the floor and is unbiased. This matches the `floor_bucketize` kernel in [`docs/modules/common/quant.md`](../../common/quant.md).

## What ADC does not own

- Clamp voltage and current-to-voltage conversion (those live in a separate clamp-driver block such as a `TIA`).
- Analog-domain rescale factors that convert ADC codes back to an ideal-integer scale.
- Column multiplexing (`AnalogMux`).

See also:

- `general.md`, `mcs_sar.md`, `sar_mono.md`
- `docs/modules/analog/tia/README.md`
- `docs/modules/common/registry_dispatch.md`
