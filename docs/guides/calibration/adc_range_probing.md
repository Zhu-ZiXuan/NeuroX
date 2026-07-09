# ADC range probing

Goal: probe the analog distribution at the ADC input of an xbar and obtain a ladder of candidate ADC input ranges, so you can pick the bracket that trades off clipping against quantisation resolution. This is the first calibration stage; the range you select here feeds the next stage, [ADC rescale calibration](adc_rescale.md).

This tool only logs (and optionally plots) range candidates. It does not choose ADC bit width, calibrate ADC codes, or solve the rescale factor.

## What it does

1. Build a physical xbar from the chip TOML with every nonideality flag `False`.
2. Replace `xbar.bl_adc` with a `ProbeADC` (capture-only stand-in).
3. Sample `weight_samples` independent programmed states in `weight_samples / batch_size` serial passes; each pass programs `batch_size` weights into the `inst_shape=(batch_size,)` xbar in parallel. For each pass: call `xbar.program(w)`, then sample `input_samples_per_weight` input vectors and broadcast them against the `batch_size` parallel weights in a single `xbar.vec_mat_mul` call (not input-chunked).
4. The `ProbeADC` accumulates $V_{\mathrm{pos}}$ / $V_{\mathrm{neg}}$ per call; the differential analog input the ADC would see is $V_{\mathrm{diff}} = V_{\mathrm{pos}} - V_{\mathrm{neg}}$.
5. After sampling, restore the original ADC and build the range-candidate ladder from the empirical $|V_{\mathrm{diff}}|$ distribution.

### `ProbeADC` — the capture-only stand-in

`ProbeADC` is a tool-local subclass of `ADC` that satisfies the inlined readout chain's static `bl_adc: ADC` interface without quantising. It:

- inherits `ADC`, so `setattr(xbar, "bl_adc", probe)` type-checks;
- is **not** registered with the ADC family (no `@ADC.register_key`) and so is never resolvable through `ADC.from_config` — it exists only to be installed manually by this tool;
- copies `max_bits` / `signed_range(...)` from the replaced ADC, and never logs dynamic events, so no per-op latency or energy state lives on the probe;
- in `convert(...)` — whose signature matches `ADC.convert`, so the inlined readout's injected `v_refs__V` reference taps land cleanly and are ignored — appends detached CPU float64 1-D copies of $V_{\mathrm{pos}}$ / $V_{\mathrm{neg}}$ to internal buffers and returns `torch.zeros_like(v_pos__V, dtype=torch.int64)`, so the downstream xbar's `vec_mat_mul` flatten chain stays valid.

A `ProbeHandle` stores the displaced original ADC outside the probe's module tree (in a `dataclass` field, not an `nn.Module`) so `xbar.modules()` is not polluted while the probe is installed. The handle is a context manager: `with install_probe_adc(xbar): ...` restores the original ADC on exit.

## The range-candidate ladder

`max_clip_rate_exp` ($N$) sets how deep the ladder goes. The tool emits one `max_abs` candidate plus one entry per power-of-10 clip rate from $10^{-2}$ down to $10^{-N}$. With `max_clip_rate_exp = 4` the ladder is `[max_abs, p99, p99.9, p99.99]`:

| Candidate | Target clip rate | Bracket |
|---|---|---|
| `max_abs` | $0$ | the widest range; $\pm \max |V_{\mathrm{diff}}|$ |
| `p99` | $10^{-2}$ | the $99$th percentile of $|V_{\mathrm{diff}}|$ |
| `p99.9` | $10^{-3}$ | the $99.9$th percentile |
| `p99.99` | $10^{-4}$ | the $99.99$th percentile |

Each candidate is reported with its **observed** clip rate alongside its **target** clip rate:

```text
ADC input range candidates:
  max_abs (clip_rate ~ 0.0000%, target 0.0000%):
    input_range__V: [-A_max, +A_max]
  p99 (clip_rate ~ 1.000%, target 1.000%):
    input_range__V: [-A_99, +A_99]
  p99.9 (clip_rate ~ 0.100%, target 0.100%):
    input_range__V: [-A_999, +A_999]
```

The **observed** clip rate is the empirical rate, $\operatorname{count}(|V_{\mathrm{diff}}| > A) / \operatorname{count}$; the **target** clip rate is the nominal $1 - q$ for the candidate's percentile $q$. Comparing the two tells you whether the sample is large enough for the percentile estimate to be trusted (see [§Sample-count safety](#sample-count-safety)).

The integer exponent $N$ in $10^{-N}$ is the single source of truth for each candidate — labels (`p99` / `p99.9` / ...) and spotlight filenames (`spotlight_exp2.png` / `spotlight_exp3.png` / ...) are derived from $N$ with no floating-point round-trip.

## CLI usage

```bash
python -m <scheme>.tools.macro_adc.statistic \
    --config <scheme>/config/macro_adc_statistic.toml \
    --plot-dir log/macro_adc/statistic/ \
    --device cuda:0
```

Workload, sampling, and plot knobs live in the TOML config; the CLI carries only runtime knobs:

| Flag | Type | Default | Role |
|---|---|---|---|
| `--config` | path | required | Statistic-run TOML; sections `[cim_macro]`, `[workload]`, `[statistic]`, `[plot]` |
| `--device` | str | `cpu` | `cpu` / `cuda` / `cuda:N`; omit to use CPU (no implicit GPU pickup) |
| `--plot-dir` | path | `None` | Optional directory; writes `overview.png` plus a per-candidate `spotlight_*.png` |
| `--log-level` | str | `INFO` | Logger level |

When `--plot-dir` is supplied, matplotlib is imported lazily; if it is not installed the tool raises `RuntimeError`.

## Output / plots

A header line precedes the candidate ladder and reports the ADC-instance pooling:

```text
adc_instance_count: 4 (pooled across all instances; identical circuits + shared bias)
```

The xbar instantiates one physically-identical `bl_adc` module per readout group. All of their captured analog inputs are flattened into a single statistical pool — the design intent, mirroring real silicon where multiple ADC slices share the same circuit design and bias network. So `adc_instance_count` is the number of identical `bl_adc` instances whose $V_{\mathrm{diff}}$ samples were merged.

When `--plot-dir DIR` is supplied the tool writes `overview.png` plus one `spotlight_*.png` per ladder entry (`spotlight_max_abs.png`, then `spotlight_exp{N}.png`). Bin widths are derived from the ADC code grid so no plot goes coarser than the LSB:

- `overview.png` — a two-panel master view. The left panel is a log-y histogram of $V_{\mathrm{diff}}$ with each candidate's $\pm A$ overlaid; its bin width is the widest candidate's LSB divided by `bins_per_code`. The right panel is a log-y complementary CDF of $|V_{\mathrm{diff}}|$ with each candidate's $A$ drawn as a dashed vertical line.
- `spotlight_*.png` — one per candidate: a histogram of $V_{\mathrm{diff}}$ whose bin width equals that candidate's LSB, so each bin spans exactly one ADC code and the bin edges align with the overlaid code-boundary grid. The candidate's $\pm A$ are drawn as bold lines and the x-axis extends slightly beyond $\pm A$ so clipped samples stay visible.

## TOML schema

```toml
[cim_macro]
_neurox_use = "1t1r_28nm.toml:cim_macro"   # path relative to this TOML

[workload]
# distribution = "..."   # omit to use a uniform synthetic workload
weight_samples = 128             # must be a multiple of batch_size
input_samples_per_weight = 16
batch_size = 128
seed = 0

[statistic]
max_clip_rate_exp = 4   # emits ladder [max_abs, p99, p99.9, p99.99]

[plot]
bits = 8                # ADC bit width driving the code-grid overlay
bins_per_code = 4       # overview hist sub-divides each ADC LSB by this much
```

`bits` and `bins_per_code` only affect plotting: `bits` sets the ADC code grid overlaid on the histograms, and `bins_per_code` sub-divides each ADC LSB so the overview never plots a bin coarser than one code.

## Sample-count safety

Estimating the bottom percentile needs roughly $10 / 10^{-N} = 10^{N+1}$ samples in the tail; with fewer, the estimate is meaningless. The tool **raises `ValueError`** when the total captured count is below $10^{N+1}$, reporting the captured count and the threshold.

So the captured sample count must satisfy

$$\operatorname{weight\_samples} \times \operatorname{input\_samples\_per\_weight} \ge 10^{(\operatorname{max\_clip\_rate\_exp} + 1)}.$$

If the constraint fails, either lower `[statistic].max_clip_rate_exp` or raise `weight_samples` $\times$ `input_samples_per_weight`.

## See also

- [Calibration overview](README.md) — where range probing sits in the calibration sequence.
- [ADC rescale calibration](adc_rescale.md) — the next stage; consumes the input range you pick here.
