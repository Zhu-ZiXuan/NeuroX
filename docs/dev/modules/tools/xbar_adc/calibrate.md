# `neurox/tools/xbar_adc/calibrate.py`

## Current role

Offline CLI that fits the scalar `rescale_factor` mapping physical ADC codes back to ideal integer VMM outputs for one configured ADC operating mode.

It does **not** choose ADC range — `--adc-mode` and the underlying range/boundary settings must already be present in the chip TOML before running this tool. Use [`statistic.md`](statistic.md) to settle the range first.

## How it works

1. Build a physical `Offset1T1RXbar` from the chip TOML with every nonideality flag `False`.
2. Build its lossless counterpart via `physical.to_ideal()`. Note: `to_ideal()` does **not** copy the programmed state — both xbars are programmed separately on every weight sample.
3. Stream `weight_samples` programmed states in `weight_samples / batch_size` serial passes; each pass programs `batch_size` weights into the `inst_shape=(batch_size,)` xbar in parallel. For each pass `w`:
    - `physical.program(w)` AND `ideal.program(w)`;
    - sample `input_samples_per_weight` input vectors, broadcast against the `batch_size` parallel weights in **a single** VMM call (not input-chunked);
    - run `phys_code = physical.vec_mat_mul(x, adc_operation_point=AdcOperationPoint(adc_mode, adc_max_bits))`;
    - run `ideal_vmm = ideal.vec_mat_mul(x, adc_operation_point=AdcOperationPoint(0, 0))` — `adc_bits=0` is the lossless sentinel that returns the raw int64 dot product;
    - accumulate `(phys_code, ideal_vmm)` pairs.
4. Apply the saturation mask (`phys_code == -2^(max_bits−1)` or `phys_code == 2^(max_bits−1) − 1`) and drop those pairs from the fit.
5. Solve `r_max = Σ p_i y_i / Σ p_i²` in float64 over the remaining pairs. Raise `ValueError` if `r_max ≤ 0` (see the `rescale_factor > 0` invariant below).
6. Derive `rescale(b) = r_max · 2^(max_bits − b)` for each `b ∈ [1, max_bits]`.

## Why fit at `max_bits` and derive the rest

An ADC mode is a fixed analog range; `adc_bits` selects the resolution within that range. The LSB scales as `2A / 2^b`, so for the same mode `rescale(b) ∝ 2^(-b)`, which gives the bit-width derivation rule above.

Fitting at the mode's max bit width is preferred because the quantization step is finest there; the LS estimate is least polluted by quantization noise. Lower bit widths are derived by power-of-two scaling rather than re-fitted independently.

This rule **requires** the ADC mode to have a uniform linear code spacing and a fixed analog range across active bit widths. ADC families that violate either must expose their own calibration semantics; this tool will not detect that mismatch.

## Saturation rule

`ADC.convert` returns signed codes in `[-2^(adc_bits−1), 2^(adc_bits−1) − 1]` (each concrete ADC owns its own conversion from native internal representation to the signed output — see [`docs/dev/modules/analog/adc/base.md`](../../analog/adc/base.md) on the convention). The saturation mask flags a sample as saturated iff `phys_code == -2^(max_bits−1)` or `phys_code == 2^(max_bits−1) − 1` — the signed extremes carry no linear-region information.

**Saturated samples are always excluded from the LS fit.** The `saturation_rate` log line is the sole diagnostic surface: a high rate signals "range too tight for workload" and the user should revisit V_ref with [`statistic.md`](statistic.md).

## `rescale_factor > 0` invariant

The fitted `r_max` is required to be strictly positive — the consumer model `M_ideal ≈ code · rescale_factor` is well-defined only when `rescale_factor > 0` (signed code carries the sign of `M_ideal` directly). If `r_max ≤ 0` after fitting, the tool raises `ValueError` with diagnostic hints (broken sign convention, misconfigured ADC, or per-mode distribution mismatch).

## CLI

Workload / sampling knobs live in a TOML config; CLI carries only
runtime knobs:

| Flag | Type | Default | Role |
|---|---|---|---|
| `--config` | path | — (required) | Calibrate-run TOML; sections `[xbar]`, `[workload]`, `[adc]` |
| `--device` | str | `cpu` | `cpu` / `cuda` / `cuda:N`; omit to use CPU (no implicit GPU pickup) |
| `--plot` | path | `None` | Optional PNG output (two-panel figure) |
| `--log-level` | str | `"INFO"` | Logger level |

### TOML schema

```toml
[xbar]
_neurox_use = "1t1r_28nm.toml:xbar"   # path relative to this TOML

[workload]
# distribution = "..."           # optional → uniform
weight_samples = 128             # must be a multiple of batch_size
input_samples_per_weight = 16
batch_size = 128
seed = 0

[adc]
mode = 0                         # operating-point index; 0 ≤ mode < xbar.adc_mode_num
```

## Output

```
ADC rescale calibration
  sample_source: synthetic
  distribution: ...
  adc_mode: 0
  calibration_bits: 4
  total_pairs: ...
  valid_pairs: ...
  saturated_pairs_excluded: ...
  saturation_rate: ...
  adc_instance_count: 4 (pooled across all instances; identical circuits + shared bias)

rescale_factor_at_max_bits: ...
rmse: ...
mae: ...
residual_mean: ...
residual_std: ...
max_abs_residual: ...

Derived rescale factors (informational; NOT calibrated, follow R(b) = R_max * 2 ** (max_bits - b) — copy into xbar config manually if needed):
  bits=4: ...  [<- calibrated]
  bits=3: ...
  bits=2: ...
  bits=1: ...

Copyable config snippet (paste under xbar.adc_calibration; calibrated entry only):
[[adc_calibration]]
adc_mode = 0
adc_bits = 4
rescale_factor = ...
```

The TOML snippet emits only the single calibrated `(adc_mode, max_bits)` record. The derived-rescale table is informational for SAR-family ADCs (which support `bits < max_bits` on the same V_ref); only the calibrated row goes into the snippet, so the user retains control over which lower-bit records (if any) to add.

For non-SAR ADCs (e.g. `GeneralADC`), the derived table is skipped entirely — those topologies don't support flexible bit widths.

The TOML snippet uses the `AdcCalibrationRecord` schema (`adc_mode`, `adc_bits`, `rescale_factor`). Paste into the `[xbar]` section of the chip TOML.

`residual_mean` materially non-zero is evidence that a scalar `rescale_factor` is insufficient to absorb the readout/reference/ADC chain offset — a higher-precision (affine) model would be needed. The tool does not fit an intercept by design.

A high `saturation_rate` indicates the ADC range is too narrow for the declared workload; revisit it with [`statistic.md`](statistic.md).

## ADC instance pooling

`Offset1T1RXbar` instantiates `n_groups` physically-identical `bl_adc` modules (one per readout group). The `(phys_code, ideal_vmm)` pairs are flattened across all of those instances before the LS fit — this is the design intent, mirroring real silicon where every ADC slice shares the same circuit design and bias network, so a single scalar `rescale_factor` is the right model.

## Plot

When `--plot PATH` is supplied:

- **Left panel** — scatter `(phys_code, ideal_vmm)` with the calibrated line `y = phys_code · r_max` overlaid. Saturated samples (always excluded from the fit) are highlighted in red.
- **Right panel** — residual histogram (`phys_code · r_max − ideal_vmm`) with `mean` and `mean ± std` lines.

matplotlib is imported lazily. If `--plot` is supplied and matplotlib is not installed, the tool raises `RuntimeError`.

## Example

```bash
python -m neurox.tools.xbar_adc.calibrate \
    --config example/config/xbar_adc_calibrate.toml \
    --plot log/xbar_adc/calibrate/mode0.png \
    --device cuda:0
```

## See also

- [`README.md`](README.md)
- [`statistic.md`](statistic.md)
- [`../logging.md`](../logging.md)
