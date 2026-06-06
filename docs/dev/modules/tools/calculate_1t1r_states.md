# `neurox/tools/calculate_1t1r_states.py`

## Current role

`calculate_1t1r_states.py` is an offline single-cell calibration CLI. It
derives the per-state RRAM conductance ladder whose 1T1R cell read
current is linearly spaced across state indices under a fixed bias.

Output is emitted through the Python `logging` framework (INFO level on `neurox.tools.calculate_1t1r_states`); the trailing two lines are paste-ready into `[xbar.core_config]`:

```toml
rram_g_max__uS = 100.0
state_to_g_map__uS = [10.0, 39.34..., 69.33..., 100.0]
```

## Layer boundary

This tool operates at the **device + one-cell circuit** level. It builds:

- one `RRAM(config, policy, inst_shape=(), dtype=float64, T__K, g_max__uS)`
- one `NMOS(config, policy, inst_shape=(), dtype=float64, T__K, W__um, L__um)`

and solves the KCL of a single cell connected as
`BL — RRAM — V_X — NMOS — SL` with `gate = WL`. It does **not**
instantiate `Xbar`, `Macro`, `Readout`, `ADC`, or any wire parasitics.

## Nonideality policy

The tool forces all randomness off via
`RRAMPolicy(prog_gamma=False, stuck_at=False, read_telegraph=False, read_thermal=False)`
and `NMOSPolicy(A_vt_mismatch=False, A_beta_mismatch=False)`.
Drift skips at `t_elapsed=0` by the model's own contract.

## CLI

All physical / design parameters are required and have no defaults.

| Flag | Type | Units / role |
|---|---|---|
| `--rram-config` | path | RRAM TOML path |
| `--rram-section` | str | TOML section key |
| `--nmos-config` | path | NMOS TOML path |
| `--nmos-section` | str | TOML section key |
| `--v-wl-V` | float | WL drive voltage [V] |
| `--v-bl-V` | float | BL drive voltage [V] |
| `--v-sl-V` | float | SL drive voltage [V] |
| `--g-max-uS` | float | RRAM design g_max [uS] |
| `--n-states` | int | State count (≥ 2) |
| `--access-nmos-W-um` | float | Access NMOS width [um] |
| `--access-nmos-L-um` | float | Access NMOS length [um] |
| `--temperature-K` | float | Operating temperature [K] |

Hard fast-fail conditions: `n_states < 2`, `g_max ≤ 0`, `V_BL ≤ V_SL`,
`W ≤ 0`, `L ≤ 0`, `T ≤ 0`.

## Numerical method

Two nested bisections, both in float64.

**Inner** — for one fixed `g`, solve KCL on `V_X`:

```text
f(V_X; g) = I_RRAM(V_BL - V_X; g) - I_NMOS(V_WL, V_X, V_SL)
```

over `V_X ∈ [V_SL, V_BL]`. `RRAM.program + RRAM.snapshot` runs once per
`g`; the resulting snapshot is reused at every inner evaluation.

**Outer** — for each target current `I_target[k]`, bisect `g ∈ [g_min, g_max]`:

```text
h(g) = I_cell(g) - I_target[k]
```

The state-zero and state-(N-1) entries are snapped to `g_min` / `g_max`
to avoid floating-point drift in the saved table.

Solver tolerances are local constants (`tol_V = 1e-12 V`,
`tol_g = 1e-9 uS`, `tol_I = 1e-9 uA`); not exposed via CLI.

## Output

Everything is emitted through `logger.info(...)` (handler set up via
`logging.basicConfig(level=logging.INFO, format="%(message)s")` in
`main()`) in the order: load notices → disabled-nonideality notices →
bias / device summary → endpoint currents → end-slope diagnostic
`dI/dg @ g_max` → target ladder → per-state derivation trace →
`max_abs_current_error__uA` → final two paste-ready lines
`rram_g_max__uS` and `state_to_g_map__uS = [...]`.

## Important non-goals

- Does not compute ADC boundaries or `v_refs__V`.
- Does not compute any `rescale_factor`.
- Does not instantiate `Xbar` / `XbarMacro` / `Readout`.
- Does not include wire parasitics (BL / SL / WL drops).
- Does not include TIA / Mux / ADC behaviour.
- Does not sample any mismatch or dynamic noise.

See also:

- `docs/dev/modules/tools/README.md`
- `docs/dev/modules/device/rram.md`
- `docs/dev/modules/device/nmos.md`
- `docs/dev/modules/xbar/_1t1r/circuit_core.md`
