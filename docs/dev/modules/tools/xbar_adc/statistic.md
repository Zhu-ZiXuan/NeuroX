# `neurox/tools/xbar_adc/statistic.py`

## Current role

Offline CLI that probes the analog distribution at the ADC input of an `Offset1T1RXbar` and recommends ADC input-range candidates.

It does **not** choose ADC bit width, calibrate ADC codes, or solve `rescale_factor`. It logs range candidates only.

## How it works

1. Build an `Offset1T1RXbar` from the chip TOML with every nonideality flag `False`.
2. Replace `xbar.readout.bl_adc` with a `ProbeADC` (capture-only stand-in).
3. Sample `weight_samples` independent programmed states in `weight_samples / batch_size` serial passes; each pass programs `batch_size` weights into the `inst_shape=(batch_size,)` xbar in parallel. For each pass:
    - `xbar.program(w)`;
    - sample `input_samples_per_weight` input vectors, broadcast against the `batch_size` parallel weights in **a single** `xbar.vec_mat_mul` call (not input-chunked).
4. The `ProbeADC` accumulates `v_pos__V` / `v_neg__V` per call; `v_diff__V = v_pos − v_neg` is the ADC's differential analog input.
5. After sampling, restore the original ADC and build a range-candidate ladder from the empirical `|v_diff|` distribution.

## `ProbeADC`

A tool-local subclass of `ADC` that satisfies the readout's static `bl_adc: ADC` interface without quantizing. It:

- inherits `ADC` (so `setattr(readout, "bl_adc", probe)` type-checks);
- is **not** registered with the ADC family (no `@ADC.register_key`) — never resolvable through `ADC.from_config`;
- copies `mode_num` / `max_bits` from the replaced ADC and reuses its bound `latency_per_op__ns` callable;
- in `convert(...)`, appends detached CPU float64 1-D copies of `v_pos__V` / `v_neg__V` to internal buffers and returns `torch.zeros_like(v_pos__V, dtype=torch.int64)` so the downstream `readout.readout(...)` flatten chain remains valid.

`ProbeHandle` stores the displaced original ADC outside the probe's module tree (in a `dataclass` field, not `nn.Module`) so `xbar.modules()` is not polluted while the probe is installed. The handle is a context manager — `with install_probe_adc(xbar): ...` restores the original on exit.

## CLI

Workload / sampling / plot knobs are baked into a TOML config (see
`example/config/xbar_adc_statistic.toml` for the production sample).
The CLI itself only carries runtime knobs:

| Flag | Type | Default | Role |
|---|---|---|---|
| `--config` | path | — (required) | Statistic-run TOML; sections `[xbar]`, `[workload]`, `[statistic]`, `[plot]` |
| `--device` | str | `cpu` | `cpu` / `cuda` / `cuda:N`; omit to use CPU (no implicit GPU pickup) |
| `--plot-dir` | path | `None` | Optional directory; writes `overview.png` + per-candidate `spotlight_*.png` |
| `--log-level` | str | `"INFO"` | Logger level |

### TOML schema

```toml
[xbar]
_neurox_use = "1t1r_28nm.toml:xbar"   # path relative to this TOML

[workload]
# distribution = "..."   # omit → uniform synthetic workload
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

## Output

### Range candidates

For each candidate (`max_abs` plus one entry per power-of-10 clip rate down to `10^-max_clip_rate_exp`):

```
ADC input range candidates:
  max_abs (clip_rate ~ 0.0000%, target 0.0000%):
    input_range__V: [-A_max, +A_max]
  p99 (clip_rate ~ 1.000%, target 1.000%):
    input_range__V: [-A_99, +A_99]
  p99.9 (clip_rate ~ 0.100%, target 0.100%):
    input_range__V: [-A_999, +A_999]
```

`clip_rate ~ X%` is the **observed** rate (empirical `count(|v_diff| > A) / count`); `target` is the nominal `1 - q`.

The integer exponent `N` in `10^-N` is the single source of truth for each candidate — labels (`p99` / `p99.9` / ...) and spotlight filenames (`spotlight_exp2.png` / `spotlight_exp3.png` / ...) are derived from `N` with no floating-point round-trip.

### ADC instance pooling

A header line reports:

```
adc_instance_count: 4 (pooled across all instances; identical circuits + shared bias)
```

`Offset1T1RXbar` instantiates `n_groups` physically-identical `bl_adc` modules (one per readout group). All of their captured analog inputs are flattened into a single statistical pool — this is the design intent, mirroring real silicon where multiple ADC slices share the same circuit design and bias network.

### Plot output

When `--plot-dir DIR` is supplied:

```
DIR/
├── overview.png            # 2-panel hist + CCDF master view
├── spotlight_max_abs.png   # widest range — observed_clip = 0%
├── spotlight_exp2.png      # p99 — clip rate 1e-2
├── spotlight_exp3.png      # p99.9 — clip rate 1e-3
└── spotlight_exp{N}.png    # one per ladder entry up to max_clip_rate_exp
```

Bin widths are **derived from the ADC code grid** so no plot ever goes
coarser than the LSB:

- **`overview.png`** (two-panel, `12 × 4 in`):
  - **Left** — log-y histogram of `v_diff__V` with each candidate's `±A`
    overlaid. Bin width = `LSB(max_abs candidate) / bins_per_code` —
    `bins_per_code = 1` plots one bin per ADC code, larger values
    reveal sub-code structure.
  - **Right** — log-y complementary CDF of `|v_diff|`; each candidate's
    `A` is overlaid as a dashed vertical line.

- **`spotlight_*.png`** — one per candidate:
  - Histogram of `v_diff__V` with bin width = `LSB(this candidate)` so
    each bin spans exactly one ADC code. Bin edges line up with the
    code-boundary lines drawn on the same axes.
  - Bold `±A` lines (candidate colour, width `1.5`, full alpha).
  - Interior code grid: `2**bits − 1` lines at `−A + k · LSB` for
    `k ∈ {1, …, 2**bits − 1}`.
  - x-axis extended `≈ 10 %` beyond `±A` at the same bin width so
    clipped samples remain visible.
  - Title: `{label}: A=±{A}V | observed_clip={x}% | bits={N} | n_codes={2**N} | LSB={2A / 2**N}V`.

matplotlib is imported lazily. If `--plot-dir` is supplied and matplotlib
is not installed, the tool raises `RuntimeError`.

## Sample-count safety

Estimating the bottom percentile requires at least roughly `10 / 10^-N = 10^(N+1)` samples in the tail; with fewer the estimate is meaningless. The tool **raises `ValueError`** when `total_captured < 10^(max_clip_rate_exp + 1)`, reporting the captured count and the threshold. Either lower `[statistic].max_clip_rate_exp` or increase `weight_samples` × `input_samples_per_weight`.

## Example

```bash
python -m neurox.tools.xbar_adc.statistic \
    --config example/config/xbar_adc_statistic.toml \
    --plot-dir log/xbar_adc/statistic/ \
    --device cuda:0
```

## See also

- [`README.md`](README.md)
- [`calibrate.md`](calibrate.md)
- [`../logging.md`](../logging.md)
