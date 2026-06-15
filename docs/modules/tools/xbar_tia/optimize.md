# `neurox/tools/xbar_tia/optimize.py`

## Current role

Offline TIA design-knob exploration. Sweeps the cartesian product of `[sweep]` axes and scores each candidate against a workload Gaussian model under a fixed `[hardware]` block. Emits a ranked summary log and optional per-slice transfer-curve plots.

## CLI

| Flag | Type | Default | Role |
|---|---|---|---|
| `--config` | path | — (required) | Tool-run TOML; sections `[hardware]`, `[workload]`, `[sweep]` |
| `--device` | str | `cpu` | Torch device |
| `--plot` | path | `None` | Optional overview PNG path |
| `--slice-plot-dir` | path | `None` | Optional directory for per-slice transfer-curve plots |
| `--top-k` | int | `10` | Number of top candidates to print |
| `--log-level` | str | `INFO` | Logger level |

## TOML schema

`OptimizeConfig` (frozen dataclass):

- `[hardware]` — chip-level constants not under sweep:
  - `nmos_config: NMOSConfig` (or `_neurox_use_preset = "process/mos:..."`).
  - `v_dd__V: float` — chip supply rail [V].
  - `v_ref__V: float` — BL clamp softclip reference voltage [V].
  - `output_saturation_softness__V: float` — softclip softness band [V].
  - `target_v_max__V: float` — user-chosen ceiling for the TIA output [V] (aligned with the ADC's largest `v_ref` mode).
  - `tia_n_newton: int` — Newton iterations for the inner TIA solve.
- `[workload]` — Gaussian model of the per-column BL current:
  - `mean__uA: float`, `std__uA: float`.
- `[sweep]` — design knobs to sweep (cartesian product):
  - `opamp_gain: list[float]`
  - `pseudo_nmos_W__um: list[float]`
  - `pseudo_nmos_L__um: list[float]`
  - `v_nmos_bias__V: list[float]`

Each field is required; the tool does not provide defaults. See [`example/config/xbar_tia_optimize.toml`](../../../../example/config/xbar_tia_optimize.toml) for a runnable template.

## Output

- Logger: ranked top-K candidates with score breakdown (`v_util`, `sat_match`, `linearity_r2`, `overshoot_safe`).
- `--plot`: overview figure (score vs candidate).
- `--slice-plot-dir`: per-slice 1D transfer curves; one PNG per axis-fixing slice through the sweep cube.
