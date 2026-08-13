# State-map calibration

Goal: derive the per-state RRAM conductance ladder `state_to_g_map__uS` so that the 1T1R cell read current is *linear in the state index* under a fixed read bias. This ladder is consumed by [reference/primitive/xbar/array/1t1r](../../reference/primitive/xbar/array/1t1r.md) (`[cim_macro.array_config.cell_config]`). The tool is the offline single-cell CLI `<scheme>/tools/calculate_1t1r_states.py`.

## What the tool produces

The CLI emits two paste-ready lines through the Python `logging` framework (INFO level on the tool's `calculate_1t1r_states` logger) for direct paste into `[cim_macro.array_config.cell_config]`:

```toml
rram_g_max__uS = 100.0
state_to_g_map__uS = [10.0, 39.34, 69.33, 100.0]
```

Entry $k$ of `state_to_g_map__uS` is the RRAM conductance whose single-cell read current equals the $k$-th rung of a current ladder spaced *uniformly* between the endpoint currents. State $0$ and state $N-1$ are snapped exactly to $g_{\min}$ and $g_{\max}$ to avoid floating-point drift in the saved table.

## Layer boundary

This tool operates at the **device + one-cell circuit** level only. It instantiates exactly:

- one `Rram` with `inst_shape = ()`, `dtype = float64`, at the configured temperature and `g_max__uS`;
- one `Nmos` with `inst_shape = ()`, `dtype = float64`, at the configured temperature, width $W$, and length $L$.

It solves the KCL of a single cell wired as

```text
BL — RRAM — V_X — NMOS — SL,    gate = WL
```

It does **not** instantiate `CimMacro`, `CimUnit`, readout blocks, or an ADC, and it includes **no** wire parasitics (no BL / SL IR drops, no WL line capacitance), boundary-clamp behaviour, muxing, or ADC behaviour. It therefore does *not* compute ADC boundaries, `v_refs__V`, or any `rescale_factor` — those belong to the ADC-side calibration guides.

## Forced all-off nonideality policy

The ladder must be deterministic, so the tool forces every randomness source off rather than reading a policy file. It builds:

- `RramPolicy(prog_gamma=False, drift=False, stuck_at=False, read_telegraph=False, read_thermal=False)`
- `MosfetPolicy(A_vt_mismatch=False, A_beta_mismatch=False)`

Conductance drift is skipped because programming uses `t_elapsed = 0.0`, which the RRAM model treats as a no-drift snap by its own contract. Each candidate conductance is applied once via `Rram.program` followed by a single `Rram.snapshot`; that snap is reused at every solver evaluation for that conductance.

## Numerical method — double nested bisection

Both loops run in float64 on CPU; there is no GPU code path for the single-cell solve.

**Inner loop (fixed conductance $g$): solve the internal node $V_X$.** For one fixed RRAM conductance $g$, find the internal node voltage $V_X$ that balances the cell KCL:

$$f(V_X; g) = I_{\mathrm{RRAM}}(V_{\mathrm{BL}} - V_X;\, g) - I_{\mathrm{NMOS}}(V_{\mathrm{WL}}, V_X, V_{\mathrm{SL}}) = 0,$$

bisecting over $V_X \in [V_{\mathrm{SL}}, V_{\mathrm{BL}}]$. The cell read current returned for this $g$ is $I_{\mathrm{cell}}(g) = I_{\mathrm{RRAM}}(V_{\mathrm{BL}} - V_X;\, g)$ evaluated at the converged $V_X$.

**Outer loop (per target current): solve the conductance $g$.** The endpoint currents $I_{\mathrm{cell}}(g_{\min})$ and $I_{\mathrm{cell}}(g_{\max})$ define a target ladder of $N$ currents $\{I_{\mathrm{target}}[k]\}_{k=0}^{N-1}$ spaced uniformly between them. For each target $I_{\mathrm{target}}[k]$, bisect $g \in [g_{\min}, g_{\max}]$ to solve:

$$h(g) = I_{\mathrm{cell}}(g) - I_{\mathrm{target}}[k] = 0.$$

Each outer evaluation of $h(g)$ runs one full inner $V_X$ bisection. The state-$0$ and state-$(N-1)$ entries are snapped to $g_{\min}$ / $g_{\max}$ rather than taken from the bisection result.

The loops fast-fail on a same-sign bracket (no root in the interval) or on failure to converge within the iteration cap, which surfaces non-monotone endpoint currents or a mis-specified bias.

### Tolerances

Convergence tolerances are local constants, not CLI flags:

| Quantity | Symbol in source | Value |
|---|---|---|
| Internal node $V_X$ | `_TOL_VX__V` | $10^{-12}$ V |
| Conductance $g$ | `_TOL_G__uS` | $10^{-9}$ uS |
| Cell current $I$ | `_TOL_I__uA` | $10^{-9}$ uA |

The run also logs `max_abs_current_error__uA` (the largest $|I_{\mathrm{cell}}(g_k) - I_{\mathrm{target}}[k]|$ across states) and a `dI/dg @ g_max` end-slope diagnostic so the linearity of the recovered ladder can be checked.

## Symbols

| Symbol | Meaning | Unit |
|---|---|---|
| $V_{\mathrm{WL}}$ | word-line drive voltage | V |
| $V_{\mathrm{BL}}$ | bit-line drive voltage | V |
| $V_{\mathrm{SL}}$ | source-line drive voltage | V |
| $V_X$ | internal RRAM-NMOS node voltage | V |
| $g$, $g_{\min}$, $g_{\max}$ | RRAM conductance (and design window) | uS |
| $I_{\mathrm{RRAM}}$ | RRAM branch current | uA |
| $I_{\mathrm{NMOS}}$ | access-NMOS branch current | uA |
| $I_{\mathrm{cell}}(g)$ | converged single-cell read current | uA |
| $I_{\mathrm{target}}[k]$ | $k$-th uniformly spaced target current | uA |
| $N$ | state count ($N \ge 2$) | — |
| $W$, $L$ | access-NMOS width / length | um |

## CLI

The tool is config-driven via the standard argument set; all physical and design parameters live in the TOML, and the CLI carries only runtime knobs.

```bash
python -m <scheme>.tools.calculate_1t1r_states --config <scheme>/config/calculate_1t1r_states.toml
```

| Flag | Type | Default | Role |
|---|---|---|---|
| `--config` | path | required | Tool-run TOML (sections `[rram]`, `[nmos]`, `[bias]`, `[design]`) |
| `--log-level` | str | `INFO` | Logger level (`DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`) |

There is intentionally no `--device` flag; the single-cell bisection solve runs entirely in float64 on CPU.

## TOML schema

The config is the frozen dataclass `Calculate1t1rStatesConfig` with four sections:

- `[rram]` — an `RramConfig` (or `_neurox_use_preset = "process/rram:..."`).
- `[nmos]` — a `MosfetConfig` (or `_neurox_use_preset = "process/mos:..."`).
- `[bias]` — the per-cell read bias used during ladder derivation:
  - `v_wl__V: float` — WL drive voltage.
  - `v_bl__V: float` — BL drive voltage.
  - `v_sl__V: float` — SL drive voltage.
  - `temperature__K: float` — operating temperature.
- `[design]` — RRAM design window, state count, and access-NMOS sizing:
  - `g_max__uS: float` — RRAM design $g_{\max}$.
  - `n_states: int` — state count ($\ge 2$).
  - `access_nmos_W__um: float` — access-NMOS width.
  - `access_nmos_L__um: float` — access-NMOS length.

Hard fast-fail conditions: $N < 2$, $g_{\max} \le 0$, $g_{\max} \le g_{\min}$, $V_{\mathrm{BL}} \le V_{\mathrm{SL}}$, $W \le 0$, $L \le 0$, $T \le 0$. A runnable template lives beside the tool at `<scheme>/config/calculate_1t1r_states.toml`.

## Output order

All output goes through `logger.info(...)` in this order: load notices $\rightarrow$ disabled-nonideality notices $\rightarrow$ bias / device summary $\rightarrow$ endpoint currents $\rightarrow$ `dI/dg @ g_max` end-slope diagnostic $\rightarrow$ target ladder $\rightarrow$ per-state derivation trace $\rightarrow$ `max_abs_current_error__uA` $\rightarrow$ the two paste-ready lines `rram_g_max__uS` and `state_to_g_map__uS`.

---

- **See also**: [XbarArray1t1r reference](../../reference/primitive/xbar/array/1t1r.md) (consumer of `state_to_g_map__uS`)
- [RRAM device reference](../../reference/primitive/device/rram.md)
- [access-NMOS device reference](../../reference/primitive/device/mosfet.md)
- [module parameter](../../conventions/module_parameter.md)
- [calibration hub](README.md)
