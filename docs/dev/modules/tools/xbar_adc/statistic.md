# `neurox/tools/xbar_adc/statistic.py`

## Current role

Offline CLI that probes the analog distribution at the ADC input of an `Offset1T1RXbar` and recommends ADC input-range candidates.

It does **not** choose ADC bit width, calibrate ADC codes, or solve `rescale_factor`. It logs range candidates only.

## How it works

1. Build an `Offset1T1RXbar` from the chip TOML with every nonideality flag `False`.
2. Replace `xbar.readout.bl_adc` with a `ProbeADC` (capture-only stand-in).
3. Sample `weight_samples` independent programmed states. For each:
    - `xbar.program(w)`;
    - sample `input_samples_per_weight` input vectors, chunked by `batch_size`, and drive each through `xbar.vec_mat_mul`.
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

| Flag | Type | Default | Role |
|---|---|---|---|
| `--xbar-config` | path | — (required) | Chip xbar TOML path |
| `--distribution` | path | `None` | Synthetic-workload distribution TOML; omitted → uniform |
| `--weight-samples` | int | `32` | Independent programmed-state samples |
| `--input-samples-per-weight` | int | `1024` | Per-weight primitive input-vector count |
| `--batch-size` | int | `256` | Per-VMM input batch cap |
| `--max-clip-rate-exp` | int | `3` | Max acceptable clip rate is `10^(-N)`, integer `N ≥ 2`. Emits ladder `[max_abs, p99, p99.9, ..., p(1-10^-N)]` |
| `--seed` | int | `None` | Optional deterministic seed |
| `--device` | str | `"auto"` | `auto` / `cpu` / `cuda` / `cuda:N` |
| `--plot-dir` | path | `None` | Optional directory; writes `overview.png` and (when `--plot-bits` is set) per-candidate `spotlight_*.png` |
| `--plot-bits` | int | — | ADC bit width for the spotlight code-grid overlay; integer in `[1, 12]`. Required when `--plot-dir` is set; rejected otherwise |
| `--log-level` | str | `"INFO"` | Logger level |

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
├── overview.png            # Always written
├── spotlight_max_abs.png   # Only when --plot-bits is also set
├── spotlight_exp2.png      # p99 — clip rate 1e-2
├── spotlight_exp3.png      # p99.9 — clip rate 1e-3
└── spotlight_exp{N}.png    # one per ladder entry up to --max-clip-rate-exp
```

**`overview.png`** (always two-panel, fixed `12 × 4 in`):

- **Left** — log-y histogram of `v_diff__V` with each candidate's `±A` overlaid as a dashed vertical pair (one colour per candidate, from `tab10`).
- **Right** — log-y complementary CDF of `|v_diff|`; each candidate's `A` is overlaid as a dashed vertical line, so you can read "for any A, what clip rate do I pay?" directly.

**`spotlight_*.png`** — written only if `--plot-bits N` is supplied (one file per candidate):

- Background: log-y histogram of `v_diff__V` in light gray.
- Bold `±A` lines: candidate colour, width `1.5`, full alpha.
- Interior code grid: `2**N − 1` lines at `−A + k · (2A / 2**N)` for `k ∈ {1, …, 2**N − 1}` in the same candidate colour, width `0.5`, alpha `0.4`.
- x-axis clamped to `[-1.1·A, +1.1·A]`; y-axis log.
- Figure width scales as `max(12, 6 + 1.2·N) in` so high-bit grids stay readable.
- Title: `{label}: A=±{A}V | observed_clip={x}% | plot_bits={N} | n_codes={2**N} | LSB={2A / 2**N}V`.

matplotlib is imported lazily. If `--plot-dir` is supplied and matplotlib is not installed, the tool raises `RuntimeError`.

## Sample-count safety

Estimating the bottom percentile requires at least roughly `10 / 10^-N = 10^(N+1)` samples in the tail; with fewer the estimate is meaningless. The tool **raises `ValueError`** when `total_captured < 10^(max_clip_rate_exp + 1)`, reporting the captured count and the threshold. Either lower `--max-clip-rate-exp` or increase `--weight-samples` × `--input-samples-per-weight`.

## Example

```bash
python -m neurox.tools.xbar_adc.statistic \
    --xbar-config neurox/presets/xbar/1t1r_28nm.toml \
    --distribution chip_workload.toml \
    --weight-samples 64 \
    --input-samples-per-weight 4096 \
    --batch-size 512 \
    --max-clip-rate-exp 4 \
    --seed 0 \
    --plot-dir out/adc_input_dist \
    --plot-bits 4
```

## See also

- [`README.md`](README.md)
- [`calibrate.md`](calibrate.md)
- [`../logging.md`](../logging.md)
