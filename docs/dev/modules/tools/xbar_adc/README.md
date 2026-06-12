# `neurox/tools/xbar_adc/`

## Current role

Offline xbar-level ADC range / calibration CLIs. Two CLIs plus a private shared backend:

| Module | Role |
|---|---|
| `statistic.py` | Probe the analog distribution at the ADC input; emit range candidates. |
| `calibrate.py` | Fit the scalar `rescale_factor` that maps physical ADC codes back to ideal VMM. |
| `_sampling.py` | Private: TOML loaders, samplers, all-off xbar builder. |
| `_probe.py` | Private: `ProbeADC` (capture-only stand-in) + `install_probe_adc` helper. |

Both CLIs are strictly xbar-facing: they operate on a single `Offset1T1RXbar` built from the chip TOML and do not touch macro, logical weights, or logical activations.

## Workflow

For each ADC operating mode `i` exposed by the chip's ADC (one mode for `GeneralADC`, `len(v_refs__V)` modes for SAR-family ADCs), the user runs the loop below:

```
chip xbar TOML (range/calibration TBD)
            |
            v
  [statistic.py]   --config xbar_adc_statistic.toml   ([statistic].max_clip_rate_exp = N)
                                                        -> per-A candidate ladder
            |
            v
  user picks a range A_i and writes the V_ref / boundary for mode i into the ADC config
            |
            v
  [calibrate.py]   --config xbar_adc_calibrate.toml   ([adc].mode = i)
                                                        -> rescale_factor for (i, max_bits)
            |
            v
  user appends one [[xbar.adc_calibration]] record (adc_mode=i, adc_bits=max_bits, rescale_factor=...)
            |
            v
  Loop until all modes are calibrated. xbar is then fully calibrated; model inference can run.
```

Neither tool mutates the TOML automatically. The user copy-pastes the recommended values; the tools only log them. Per-mode workload assumptions go in `[workload].distribution` (relative to the run TOML) — different modes typically target different signal regions and need different distributions, so each gets its own run TOML.

If the user wants to run an ADC at `adc_bits < max_bits`, the lower-bit `rescale_factor` is derived from the `max_bits` calibrated value by `R(b) = R_max · 2^(max_bits − b)`. `calibrate.py` prints this derived table as informational output (SAR families only); the user adds the lower-bit records by hand.

## Synthetic-workload distribution

Both CLIs accept an optional `[workload].distribution` path in the run TOML (relative to the run TOML). Schema:

```toml
[w]
values = [-1, 0, 1]
probs = [0.20, 0.65, 0.15]

[x]
values = [0, 1, 2, 3]
probs = [0.70, 0.20, 0.08, 0.02]
```

- `[w]` describes the per-cell programmed-digit distribution. Values must lie in `xbar.w_digit_range`.
- `[x]` describes the per-row primitive input code distribution. Values must lie in `xbar.x_range`.
- Either section may be absent — the missing axis falls back to uniform sampling.
- `probs` need not sum to 1; the tool normalises internally.

## Scope

- **Supported readout topology**: `OffsetSwitchCapMuxAdcReadOut` only. Other readouts raise `TypeError`.
- **Supported ADC families** (for the all-off policy builder): `GeneralADC`, `SarAdcMono`, `McsSarAdc`. Other ADC config types raise `TypeError`.
- **Determinism**: `[workload].seed` in the run TOML; with a seed set the run is reproducible on the same device.
- **Device**: `--device cpu|cuda|cuda:N`; omit to use CPU. The resolved device is logged.

## Relation to other tools

- [`calculate_1t1r_states.py`](../calculate_1t1r_states.md) calibrates the single-cell RRAM conductance ladder — runs at the device + one-cell layer, not the xbar layer.

## See also

- [`statistic.md`](statistic.md)
- [`calibrate.md`](calibrate.md)
- [`../logging.md`](../logging.md)
