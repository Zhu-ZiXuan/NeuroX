# NMOS Device Model — Design

This document describes the physical NMOS model in
[`neurox/device/nmos.py`](../../neurox/device/nmos.py).  The class is a general
electrical primitive: it consumes node voltages and reports drain–source
current and its analytical derivatives, with no assumption about which
terminal is the source vs. drain or which region of operation the device sits
in.  The 1T1R xbar is one consumer; future peripheral-CMOS models (inverters,
sense amplifiers, latches) are another.

## 1. Why a physical model

`NMOSConfig` exposes **PDK physical parameters** directly — geometry (`W`,
`L`, source/drain areas and perimeters) and process constants (`μ·Cox`,
`V_th`, `n_factor`, `T`, gate-oxide / overlap / junction capacitance
densities).  Every macro-level quantity callers need (β, parasitic caps, the
small-signal conductances) is derived from these inputs by closed-form
physics inside `NMOS.__init__` or `NMOS.fabricate` — no manual recomputation,
and "what if W doubles?" is a single config edit away.

## 2. Continuous EKV-softplus I-V

We use the EKV symmetric formulation with a softplus smoothing of the
overdrive voltage.  Letting `η = n_factor · V_T` and
`smooth_scale = 2 · η`,

```
V_eff,s = softplus(V_g − V_s − V_th0,  β = 1 / smooth_scale)
V_eff,d = softplus(V_g − V_d − V_th0,  β = 1 / smooth_scale)
I_ds    = ½ · β · (V_eff,s² − V_eff,d²)
```

Limits:

| Region | `softplus(V_ov)` reduces to | Resulting `I_ds` |
|---|---|---|
| Strong inversion (`V_ov ≫ 0`) | `V_ov` | square-law `½·β·V_ov²` |
| Subthreshold (`V_ov ≪ 0`)     | `2η·exp(V_ov/2η)` | exponential diffusion `∝ exp(V_ov/η)` |
| Crossover (`V_ov ≈ 0`)        | continuous, twice-differentiable transition | continuous |

Both regions are captured by one analytic expression — the same code path
serves a fully-on access transistor and a fully-off subthreshold leaker.

## 3. Analytical derivatives

Because `d softplus(x)/dx = sigmoid(x)`, every conductance is closed form.
With `σ_s = sigmoid((V_g − V_s − V_th0) / smooth_scale)` and the analogous
`σ_d`:

```
g_ds = ∂I_ds / ∂V_ds = ½ · β · (V_eff,d · σ_d  +  V_eff,s · σ_s)   (Vcm fixed)
g_m  = ∂I_ds / ∂V_g  =      β · (V_eff,s · σ_s  −  V_eff,d · σ_d)
```

`g_ds` is the symmetric Vcm-fixed drain–source small-signal conductance —
the same quantity an MNA solver would stamp between drain and source nodes.
In deep triode (`V_ds ≈ 0`, `V_ov ≫ 0`) it collapses to `β · (V_gs − V_th)`
— the classical "g_on".  In strong subthreshold it collapses to a value
proportional to `β · η · exp((V_g − V_s − V_th0) / η)` — the diffusion
conductance that used to be modelled separately as `g_off`.

## 4. Fabrication and buffer lifecycle

Two pairs of non-persistent buffers live on the module:

* `nominal_beta__uA_per_V2` / `nominal_vth__V` — scalar PDK targets,
  derived at `__init__` from cfg + temperature.  Always present.
* `beta__uA_per_V2` / `vth__V` — per-cell fabricated buffers,
  registered at `fabricate(shape)` time.  Empty placeholders until
  then.

The `nominal_` prefix reserves the unprefixed name for the per-cell
sampled value — anything called `beta__uA_per_V2` after fabricate is
a per-cell tensor.

```python
nmos.fabricate(shape=(..., sign, phys_col, row))
```

independently draws Gaussian mismatch on (W, L, μ·Cox, V_th) per the
configured `A_vt__mV_um` / `A_beta_relative__um` sigmas, computes
per-cell `β = μ·Cox · W / L`, and **registers** both buffers at the
requested shape.  No return value — the buffers stay inside the
module.  After the call, `solve_dc` broadcasts the internal buffers
against per-cell node voltages naturally.

The buffers are non-persistent: they represent a fabricated chip's
sampled fingerprint and are not part of the algorithm-level
`state_dict`.  Each fabricated NMOS instance is owned by exactly one
circuit (typically a `Core1T1R` or a TIA), and per
`temp/state_holding.md` is not shared with another core.

Per-VMM dynamic state goes through
`nmos.snapshot(...) → NMOSSnapshot` instead.  Today the
runtime contract is empty; the call is kept as a sealed entry point
so future dynamic-noise additions plug into the existing solver
loop without API churn.

## 5. Parasitic capacitances

The four lumped caps are compiled from geometry + PDK in `NMOS.__init__`
and carried as Python-float scalars so the hot path sees zero tensor work:

| Cap | Formula | Terminals |
|---|---|---|
| `c_gs__fF` | `0.5 · W·L·C_ox + W·C_gso` | Gate ↔ Source |
| `c_gd__fF` | `0.5 · W·L·C_ox + W·C_gdo` | Gate ↔ Drain |
| `c_db__fF` | `A_D·C_j + P_D·C_jsw`     | Drain ↔ Bulk (ground) |
| `c_sb__fF` | `A_S·C_j + P_S·C_jsw`     | Source ↔ Bulk (ground) |

The 50/50 `W·L·C_ox` split is the exact deep-triode partitioning of the
intrinsic channel charge.  Overlap caps are per-unit-width; junction caps
split into bottom-area and sidewall-perimeter components.  See
[`xbar_1t1r_energy.md`](xbar_1t1r_energy.md) for how these aggregate into
the per-row / per-node / per-cell scalars the energy model uses.

## 6. Process variation

Per-cell mismatch is parameterised in the standard Pelgrom form: the
matching coefficient is set in the config and the per-cell σ is derived
from the nominal area at init time.

| Field | Physical origin | σ formula |
|---|---|---|
| `A_vt__V_um` | random dopant fluctuation, work-function variation | `σ_Vt = A_vt / √(W·L)` |
| `A_beta_relative__um` | combined W, L, μ, C_ox fluctuations | `σ_β / β = A_beta_relative / √(W·L)` |

Each coefficient is optional — `None` skips that mismatch stage.  Both
sigmas are computed once in `NMOS.__init__` from the nominal `W·L` and
reused at every `fabricate(shape)` call (per-cell area is uniform).
Mismatches collapse the four-parameter (W, L, μ, C_ox, V_th) Gaussian
sweep of older models into the two PDK-supplied Pelgrom coefficients,
which is what foundry datasheets actually report.

## 7. 1T1R integration

The 1T1R xbar consumes the device directly — there is no pre-baked
switch tensor.  At fabricate time:

1. `Core1T1R.__init__` instantiates `self.nmos = nmos_factory()`
   (per-core, not shared) and reads `V_DD,WL = wl_dac.code_to_signal[1]`.
2. `Core1T1R.fabricate(w_phys)` calls `self.nmos.fabricate(w_phys.shape)`
   to populate the per-cell `β` / `V_th` buffers inside the NMOS
   submodule.

In the forward path (`Core1T1R.forward`):

3. The core samples a per-VMM runtime snapshot via
   `nmos.snapshot(shape=full_shape)` and computes the per-cell
   gate voltage tensor `vg__V = wl_logic · V_DD,WL`.
4. It hands `nmos`, `vg__V`, the runtime snapshot, and the
   clamp driver's `clamp_snapshot` to `NewtonRaphsonSolver1T1R.solve`,
   which calls `nmos.solve_dc(vg, v_x, v_sl, nmos_snapshot)` for the
   bundled current + partials at every iteration.

This keeps the solver self-consistent: deep-triode and subthreshold come
out of the same continuous expression, evaluated at the actual converged
operating point rather than a pre-computed secant.

## 8. 28 nm reference values

See the `[nmos]` table in
[`neurox/config/default_1t1r.toml`](../../neurox/config/default_1t1r.toml)
for a complete 28 nm HKMG RVT parameter set.
