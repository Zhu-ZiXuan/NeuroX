# TIA design exploration

Goal: pick the trans-impedance amplifier (TIA) design knobs — op-amp gain, pseudo-NMOS load geometry, and NMOS bias — that best read out one xbar tile's bit-line current, given a Gaussian model of the per-column workload current. This is an offline, config-driven sweep: it does not touch a model or a checkpoint, only a TIA circuit and a current distribution.

The tool is `neurox.tools.xbar_tia.optimize`. It sweeps the cartesian product of four design axes under one fixed `[hardware]` block, scores each candidate against the `[workload]` Gaussian, and prints a ranked top-K plus optional transfer-curve plots. Everything comes from a single tool-run TOML — there is no chip preset on the command line and no other flag carries design state.

## How it works

For each point in the sweep grid the tool runs the same four shared primitives (in `neurox/tools/xbar_tia/_common.py`):

1. `build_tia(config, *, device)` — assemble an `OpAmpTIA` from the `[hardware]` block plus the one sweep cell (gain, W, L, bias).
2. `sweep_transfer(tia, *, i_min_uA, i_max_uA, n_points, device)` — evaluate the TIA transfer curve $v_\mathrm{out}(I_\mathrm{BL})$ over an input bit-line current range, returning a `TransferCurve`.
3. `fit_to_workload(curve, *, mean_uA, std_uA)` — derive workload statistics against the curve: the output voltage at $\mu$ and at $\mu \pm 3\sigma$, the slope at $\mu$, the saturation onset, and the linearity over the useful range.
4. `linearity_r2(curve, *, lo_uA, hi_uA)` — the coefficient-of-determination ($R^2$) linearity score of $v_\mathrm{out}$ vs $I_\mathrm{BL}$ inside the workload band.

### Scoring

Higher is better. Each candidate is scored against the `[workload]` Gaussian, and the logger reports a score breakdown over its terms:

- `linearity_r2` — a linearity term: how straight the readout is over the range the workload uses.
- `v_util` / `range_use` — an output-range utilisation term: how much of the chosen output ceiling the workload band fills.
- `sat_match` / `overshoot_safe` — a saturation/overshoot-guard term: it penalises a TIA whose readout range fails to cover the high-current tail or overshoots the output ceiling toward the chip rail.

The intent is a TIA that is linear where it matters, fills the chosen output ceiling, and does not clip the high-current tail. The exact objective is defined in the tool source (`neurox/tools/xbar_tia/optimize.py`) and is the authority.

## How to run

```bash
python -m neurox.tools.xbar_tia.optimize \
    --config example/config/xbar_tia_optimize.toml \
    --device cpu \
    --top-k 10
```

### CLI flags

| Flag | Type | Default | Role |
|---|---|---|---|
| `--config` | path | required | Tool-run TOML with `[hardware]`, `[workload]`, `[sweep]` sections |
| `--device` | str | `cpu` | Torch device |
| `--plot` | path | `None` | Optional overview PNG (top-K curves vs score) |
| `--slice-plot-dir` | path | `None` | Optional directory for per-slice 1D transfer-curve PNGs |
| `--top-k` | int | `10` | Number of top candidates to print |
| `--log-level` | str | `INFO` | Logger level |

### Output

- Logger: the ranked top-K candidates, each with its raw $(\mathrm{gain}, W, L, v_\mathrm{bias})$ tuple and the score breakdown — `linearity_r2`, `v_util`, `sat_match`, `overshoot_safe`, the output voltages at $\mu - 3\sigma$, $\mu$, $\mu + 3\sigma$, and the slope at $\mu$ in mV/uA.
- `--plot`: one overview figure overlaying the top-K transfer curves against the workload band.
- `--slice-plot-dir`: per-slice 1D transfer curves — one PNG per axis-fixing slice through the four-dimensional sweep cube (e.g. fix gain, L, bias and vary W).

Nothing else is written; the tool emits no config or checkpoint.

## Config schema

The config maps to a frozen `TiaDesignConfig` (`neurox/tools/xbar_tia/optimize.py`) with three required sections. There are no defaults — every field must be present.

```toml
[hardware]
v_dd__V = 0.9                          # chip supply rail [V]
v_ref__V = 0.1                         # BL clamp softclip reference voltage [V]
output_saturation_softness__V = 0.45   # softclip softness band [V]
target_v_max__V = 0.8                  # user-chosen TIA output ceiling [V]
tia_n_newton = 5                       # Newton iterations for the inner TIA solve

[hardware.nmos_config]
_neurox_use_preset = "process/mos:nmos_28_rvt"

[workload]
mean__uA = 170.26                      # per-column BL current mean [uA]
std__uA = 29.19                        # per-column BL current std [uA]

[sweep]
opamp_gain        = [30]
pseudo_nmos_W__um = [0.15, 0.18, 0.21]
pseudo_nmos_L__um = [0.03]
v_nmos_bias__V    = [0.90]
```

- `[hardware]` — chip-level constants held fixed across the whole sweep. `target_v_max__V` is the user-chosen ceiling for the TIA output, aligned with the ADC's largest `v_ref` mode, so $v_\mathrm{out}$ should fill $[0, \mathrm{target\_v\_max}]$ rather than the full rail $[0, v_\mathrm{dd}]$. The embedded `[hardware.nmos_config]` table is an `NMOSConfig` and may use the `_neurox_use_preset` directive to point at a process file.
- `[workload]` — the Gaussian model $\mathcal{N}(\mu, \sigma^2)$ of the per-column bit-line current. Derive $\mu$ and $\sigma$ from the chip's conductance state map and the activation statistics; the bundled example shows the full derivation in its comments.
- `[sweep]` — the four design axes. The tool sweeps their cartesian product, so $|\mathrm{gain}| \times |W| \times |L| \times |v_\mathrm{bias}|$ candidates total; each axis must be non-empty.

A runnable template lives at `example/config/xbar_tia_optimize.toml`.

## See also

- [Reference: op-amp TIA](../../reference/analog/tia/opamp_tia.md) — the device physics and transfer model the sweep evaluates.
- [Reference: ADC base](../../reference/analog/adc/base.md) — the readout stage whose `v_ref` mode sets `target_v_max__V`.
- [Internals: op-amp TIA](../../internals/analog/tia/opamp_tia.md) — the `OpAmpTIA` Newton solve invoked by `build_tia`.
- [API: configuration](../../api/configuration.md) — the TOML schema and `_neurox_use_preset` directive.
- [Reference: parameter provenance](../../reference/parameter_provenance.md) — sourcing the `[workload]` and `[hardware]` numbers.
- [Calibration guides](../calibration/README.md) — the complementary task of fitting model parameters to a chip.
- [Guides index](../README.md)
