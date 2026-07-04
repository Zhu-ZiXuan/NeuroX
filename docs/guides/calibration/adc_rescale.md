# ADC rescale calibration

Goal: fit the scalar `rescale_factor` that maps a physical ADC's signed integer codes back to the ideal integer VMM output, for one configured ADC operating point. This is the second calibration stage: the analog range must already be settled by [ADC range probing](adc_range_probing.md) — this tool does not choose the range, it only calibrates the code-to-output scale for a range that is already fixed in the chip TOML.

The fitted record is a single `(adc_mode, adc_bits) → rescale_factor` entry. The consumer model is $M_{\mathrm{ideal}} \approx \mathrm{code} \cdot \mathrm{rescale\_factor}$, where the signed code carries the sign of $M_{\mathrm{ideal}}$ directly.

## Prerequisite

The ADC mode and its underlying range / boundary settings must already be present in the chip TOML before running this tool — `--adc-mode` selects an existing operating point, it does not create one. Settle the range first with [ADC range probing](adc_range_probing.md); a high saturation rate reported here (see below) means the range is still too tight and you must return to that stage.

## Methodology

The fit drives a physical tile and its lossless twin against the same inputs, then solves a one-parameter least squares.

1. Build a physical xbar from the chip TOML with every nonideality flag `False`.
2. Build its lossless counterpart via `to_ideal()`. Note: `to_ideal()` does **not** copy the programmed state — both xbars are programmed separately on every weight sample.
3. Stream `weight_samples` programmed states in `weight_samples / batch_size` serial passes; each pass programs `batch_size` weights into the `inst_shape=(batch_size,)` xbar in parallel. For each pass $w$:
    - call `physical.program(w)` **and** `ideal.program(w)` as separate program calls (the twin shares no state with the physical tile);
    - sample `input_samples_per_weight` input vectors, broadcast against the `batch_size` parallel weights in a single VMM call (not input-chunked);
    - run `phys_code = physical.vec_mat_mul(x, adc_operation_point=AdcOperationPoint(adc_mode, adc_max_bits))`;
    - run `ideal_vmm = ideal.vec_mat_mul(x, adc_operation_point=AdcOperationPoint(0, 0))` — `adc_bits=0` is the lossless sentinel that returns the raw int64 dot product;
    - accumulate the $(\mathrm{phys\_code}, \mathrm{ideal\_vmm})$ pairs.
4. Apply the saturation mask and drop the flagged pairs from the fit (see [Saturation filtering](#saturation-filtering)).
5. Solve the rescale at max bit width on the surviving pairs (see [Least-squares fit](#least-squares-fit)).
6. Derive the lower-bit-width factors by power-of-two scaling (see [Bit-width derivation](#bit-width-derivation)).

## Saturation filtering

`ADC.convert` returns signed codes in $[-2^{b-1},\ 2^{b-1} - 1]$ for `adc_bits` $= b$ (each concrete ADC owns its own conversion from native internal representation to the signed output). A pair is flagged as saturated iff

$$\mathrm{phys\_code} = -2^{(\mathrm{max\_bits} - 1)} \quad \text{or} \quad \mathrm{phys\_code} = 2^{(\mathrm{max\_bits} - 1)} - 1.$$

The signed extremes carry no linear-region information, so **saturated samples are always excluded from the least-squares fit**. The reported `saturation_rate` is the diagnostic surface: a high rate signals "range too tight for workload", and you should revisit the analog range with [ADC range probing](adc_range_probing.md).

## Least-squares fit

The rescale at max bit width is the one-parameter least-squares estimate over the surviving pairs $(p_i, y_i) = (\mathrm{phys\_code}_i, \mathrm{ideal\_vmm}_i)$, computed in float64:

$$r_{\max} = \frac{\sum_i p_i y_i}{\sum_i p_i^2}.$$

If $r_{\max} \le 0$ after fitting, the tool raises `ValueError`. This enforces the `rescale_factor > 0` invariant: the consumer model $M_{\mathrm{ideal}} \approx \mathrm{code} \cdot \mathrm{rescale\_factor}$ is well-defined only when $\mathrm{rescale\_factor} > 0$, since the signed code is required to carry the sign of $M_{\mathrm{ideal}}$ directly. A non-positive estimate indicates a broken sign convention, a misconfigured ADC, or a per-mode distribution mismatch.

## Bit-width derivation

An ADC mode is a fixed analog range; `adc_bits` selects the resolution within that range. For a range half-width $A$, the LSB scales as $2A / 2^b$, so for the same mode $\mathrm{rescale}(b) \propto 2^{-b}$. The factor at each bit width $b \in [1, \mathrm{max\_bits}]$ is therefore derived from the single fit:

$$\mathrm{rescale}(b) = r_{\max} \cdot 2^{(\mathrm{max\_bits} - b)}.$$

The fit is performed **at `max_bits`** because the quantization step is finest there, so the least-squares estimate is least polluted by quantization noise; lower bit widths are obtained by power-of-two scaling rather than re-fitted independently. This rule **requires** the ADC mode to have uniform linear code spacing and a fixed analog range across active bit widths. ADC families that violate either must expose their own calibration semantics; this tool will not detect that mismatch.

## CLI

Workload and sampling knobs live in a TOML config; the CLI carries only runtime knobs.

| Flag | Type | Default | Role |
|---|---|---|---|
| `--config` | path | required | Calibrate-run TOML; sections `[xbar]`, `[workload]`, `[adc]` |
| `--device` | str | `cpu` | `cpu` / `cuda` / `cuda:N`; omit for CPU (no implicit GPU pickup) |
| `--plot` | path | `None` | Optional PNG output (two-panel figure) |
| `--log-level` | str | `"INFO"` | Logger level |

```toml
[xbar]
_neurox_use = "1t1r_28nm.toml:xbar"   # path relative to this TOML

[workload]
# distribution = "..."           # optional; defaults to uniform
weight_samples = 128             # must be a multiple of batch_size
input_samples_per_weight = 16
batch_size = 128
seed = 0

[adc]
mode = 0                         # operating-point index; 0 <= mode < xbar.adc_mode_num
```

```bash
python -m <scheme>.tools.xbar_adc.calibrate \
    --config <scheme>/config/xbar_adc_calibrate.toml \
    --plot log/xbar_adc/calibrate/mode0.png \
    --device cuda:0
```

## Output

The tool reports the fitted `rescale_factor_at_max_bits` plus residual diagnostics (`rmse`, `mae`, `residual_mean`, `residual_std`, `max_abs_residual`) and the sample accounting (`total_pairs`, `valid_pairs`, `saturated_pairs_excluded`, `saturation_rate`); `adc_instance_count` is the pooled count across all readout instances (see below). A `residual_mean` that is materially non-zero is evidence that a scalar `rescale_factor` cannot absorb the readout / reference / ADC chain offset; an affine (intercept) model would be needed, which this tool does not fit by design.

```text
ADC rescale calibration
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
```

It emits a copyable TOML snippet for the single calibrated `(adc_mode, max_bits)` record only. The snippet uses the `AdcCalibrationRecord` schema (fields `adc_mode`, `adc_bits`, `rescale_factor`):

```toml
[[adc_calibration]]
adc_mode = 0
adc_bits = 4
rescale_factor = ...
```

For SAR-family ADCs (which support `bits < max_bits` on the same range) the derived lower-bit table is printed for information; you choose which derived rows to add to the chip TOML. For non-SAR ADCs (e.g. `GeneralADC`) the derived table is skipped, as those topologies do not support flexible bit widths. Paste the snippet under the `[xbar]` section of the chip TOML.

A scheme xbar that instantiates one physically-identical `bl_adc` module per readout group flattens the $(\mathrm{phys\_code}, \mathrm{ideal\_vmm})$ pairs across all instances before the fit. This is intentional — real silicon shares the same ADC circuit design and bias network across slices, so a single scalar `rescale_factor` is the correct model.

When `--plot PATH` is supplied, the left panel scatters $(\mathrm{phys\_code}, \mathrm{ideal\_vmm})$ with the calibrated line $y = \mathrm{phys\_code} \cdot r_{\max}$ overlaid (saturated, excluded samples in red), and the right panel is the residual histogram $\mathrm{phys\_code} \cdot r_{\max} - \mathrm{ideal\_vmm}$ with mean and $\mathrm{mean} \pm \mathrm{std}$ lines. matplotlib is imported lazily; if it is missing the tool raises `RuntimeError`.

## See also

- [ADC range probing](adc_range_probing.md) — the prerequisite stage that settles the analog range this tool calibrates against.
- [ADC base reference](../../reference/analog/adc/base.md) — the signed code convention and the `convert` contract.
- [Calibration hub](README.md) — all calibration stages.
