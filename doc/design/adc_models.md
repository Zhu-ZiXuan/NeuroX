# ADC Family — Physics-Based Subclasses

NeuroX ships seven concrete ADC topologies under `neurox.analog.adc`,
all implementing the abstract `ADC` interface in
[`neurox/analog/adc/base.py`](../../neurox/analog/adc/base.py).  The
xbar holds any of them through one handle and the macro derives
`output_rescale_factor` automatically from the active operating point.

## Why a family

The previous single-class `GeneralADC` was a behavioural
boundary-bucketize.  Real chips' accuracy / energy / latency vary
sharply by topology — a SAR ADC takes `(N+1) * clk` cycles and
dissipates capacitor-DAC energy; a pipeline trades depth for
throughput; a ramp ADC has minimal area but `2**N` worst-case
latency.  Studying these trade-offs requires distinct subclasses,
not a flag.

## Runtime multi-mode contract

Every subclass is **runtime multi-mode**: `convert`,
`rescale_factor`, and `latency_per_op__ns` all take explicit `mode`
and `bits` keyword arguments at every call site.

| Method / property | Returns | Notes |
|---|---|---|
| `convert(v_pos__V, v_neg__V, *, mode, bits) -> (code, energy)` | `(int, float)` | Differential converter.  Code dtype is `int16` for boundary-bucketize variants and `int32` for SAR; energy is per-element float. |
| `rescale_factor(*, mode, bits)` | `float` | ADC-native rescale ratio between max-precision config and the active `(mode, bits)`.  Single-mode subclasses return `1.0` at their only legal point. |
| `latency_per_op__ns(*, bits)` | `float` (ns) | Latency at the active bit width.  No `mode` kwarg — every modelled topology has V_ref-independent latency. |
| `fabricate()` | `None` | Sample static per-instance state.  Default no-op; SAR variants register cap mismatch + comparator offset. |
| `area_per_inst__um2`, `leakage_per_inst__uW` | `float` | Standard PPA. |

`ADCMode = (n_bits, n_states, max_signal)` is the per-mode descriptor
used by the flash-style configs to anchor uniform-boundary derivation;
SAR uses its own `(max_bits, v_refs__V)` config and does not consume
`ADCMode`.

Floor semantics: boundaries are placed at code edges (`B_C = C · LSB`),
not midpoints (`(C - 0.5) · LSB`).  Stochastic rounding adds
`uniform(0, LSB)` jitter before the floor and is unbiased.

## Mode / bits convention

* **Single-mode behavioural ADCs** (`GeneralADC`, `PipelineADC`,
  `CyclicADC`, `RampADC`) accept only `mode == 0` and
  `bits == max_bits` (the calibrated grid).  Other values raise
  `ValueError`.  These topologies share a single comparator network
  and don't model runtime reconfiguration.
* **Multi-mode SAR** (`McsSarAdc`) supports `mode ∈ [0, len(v_refs__V))`
  and `bits ∈ [1, max_bits]`.  Lower `bits` shortens the SAR loop;
  alternate `mode` selects a smaller V_ref (rescaling LSB).

The xbar passes the active `(adc_mode, adc_bits)` from its config to
every ADC call.  See `[xbar]` in `default_1t1r.toml` for the runtime
wiring.

## Topologies

### `GeneralADC` — boundary-bucketize fallback

`neurox.analog.adc.general` — arbitrary sorted threshold list, three
optional Gaussian noise stages, constant per-op energy / latency.
Useful for calibration work (boundaries fitted by
`neurox.tools.xbar_adc_boundaries`) and as a behavioural baseline
when no specific topology is being studied.  Implied bit width:
`ceil(log2(len(boundaries) + 1))`.

### `McsSarAdc` — V_cm-based (MCS) differential SAR

`neurox.analog.adc.mcs_sar` — capacitor-DAC + comparator with
bottom-plate sampling and Merged Capacitor Switching.  Latency
`(bits + 1) · clk`.  Energy is sample + per-cycle MCS switching +
reset + lumped bootstrap / comparator / logic overheads.  Carries
two independent Pelgrom-mismatched cap arrays (one per leg) and a
static comparator offset, sampled in `fabricate()`.

### `SarAdcMono` — monotonic (Set-and-Down) differential SAR

`neurox.analog.adc.sar_mono` — placeholder.  Fabricate path is
wired, but `convert()` raises `NotImplementedError` pending the
differential Set-and-Down kernel.  Use `McsSarAdc` for SAR work.

### `PipelineADC`

`neurox.analog.adc.pipeline` — N residual-amp stages, throughput one
sample/clk after fill.  Latency `(n_stages + pipeline_depth) · clk`.

### `CyclicADC`

`neurox.analog.adc.cyclic` — single-stage feedback loop reused N
times.  Latency `n_bits · clk`.  Smaller area than pipeline at lower
throughput.

### `RampADC`

`neurox.analog.adc.ramp` — counter + comparator scanning a linear
reference.  Latency `2 ** n_bits · clk` (worst case).

## Multi-mode operating points

SAR shares a single physical capacitor / comparator array but
exposes several `(V_ref, bits)` operating points selected at
runtime.  Example configuration:

```python
McsSarAdcConfig(
    max_bits=8,
    v_refs__V=(1.2064, 0.4000, 0.2000),  # strictly decreasing
    clk_period__ns=1.0,
    c_unit__fF=1.0,
    cap_mismatch_sigma_relative=0.01,
    ...
)
```

Mode 0 is the highest-precision; lower modes share the same cap
network but use a smaller V_ref.  The xbar selects the runtime
operating point via `adc_mode` / `adc_bits` in its config, and the
macro picks up the corresponding `rescale_factor(mode, bits)`
automatically.

## Stochastic rounding

When `module.training` is True (or `config.stochastic` is forced True)
each `convert()` adds `uniform(0, LSB)` jitter before the floor.  This
yields unbiased gradients under coarse output grids — see
[`hat_training.md`](hat_training.md) for the calibration-then-freeze
rationale.

`module.eval()` (the production HAT path) disables jitter; the same
boundary list bucketizes deterministically.

## Calibration

Boundaries for the highest-precision mode come from
`neurox.tools.xbar_adc_boundaries` — see
[`calibration.md`](calibration.md).
