# 1T1R Cell

## Summary

The 1T1R cell is the concrete realization of the [cell family contract](../cell.md): the two-terminal element the array solver sees between a bit-line node and a source-line node. It models one RRAM device in series with an access NMOS, $\mathrm{BL} - \mathrm{RRAM} - V_{\mathrm{X}} - \mathrm{NMOS} - \mathrm{SL}$, with the NMOS gate driven by the word line. To the solver it is a single condensed branch: the cell solves its own internal access node $V_{\mathrm{X}}$ and exposes one branch current with two signed terminal conductances. This document specifies the cell's physical model, the access-node condensation, the signed-conductance contract it presents to the [solver](solver.md), and its device-capacitor energy; the array-level wire ladder, boundaries, and array energy are in [circuit_core](circuit_core.md).

## Physical model

The cell has one internal node, the access node $V_{\mathrm{X}}$ between the RRAM and the access NMOS. The RRAM conducts between the bit line $V_{\mathrm{BL}}$ and $V_{\mathrm{X}}$; the access NMOS conducts between $V_{\mathrm{X}}$ and the source line $V_{\mathrm{SL}}$, gated by the word-line voltage $V_{\mathrm{WL}}$. The EKV access-NMOS model is source/drain symmetric, so source and drain labels are a naming convention: the SL-side terminal ($V_{\mathrm{SL}}$) is taken as the source and the internal BL-side terminal ($V_{\mathrm{X}}$) as the drain, with no effect on the device current. The word-line drive is an input to the cell, carried in the per-call snapshot together with the device read state.

The cell presents a **two-terminal branch** to the array solver. The internal node $V_{\mathrm{X}}$ is condensed away inside the cell, so the array solver never sees it: the cell is one element between $V_{\mathrm{BL}}$ and $V_{\mathrm{SL}}$ whose current and terminal conductances summarize the entire series stack. The RRAM and NMOS device transfer functions $I_{\mathrm{R}}(\cdot)$ and $I_{\mathrm{N}}(\cdot)$ are specified in [reference/device](../../device/README.md).

## Governing equations

At a fixed terminal pair $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ the access node satisfies the internal KCL — the RRAM current into $V_{\mathrm{X}}$ equals the NMOS current out of it,

$$F_{\mathrm{X}} = I_{\mathrm{N}}\!\left(V_{\mathrm{WL}}, V_{\mathrm{X}}, V_{\mathrm{SL}}\right) - I_{\mathrm{R}}\!\left(V_{\mathrm{BL}} - V_{\mathrm{X}}\right) = 0.$$

Solving this for $V_{\mathrm{X}}$ condenses the cell. The condensed branch then reports a **single consistent current**: at convergence $I_{\mathrm{R}} = I_{\mathrm{N}}$, and that common value is the branch current $I$ (positive from $V_{\mathrm{BL}}$ to $V_{\mathrm{SL}}$). The residual $\lvert I_{\mathrm{N}} - I_{\mathrm{R}} \rvert$ at the returned $V_{\mathrm{X}}$ is the per-cell convergence diagnostic.

The cell also returns the two **signed terminal conductances** the array wire Newton needs, obtained by eliminating $V_{\mathrm{X}}$ from the series stack — the series condensation of the NMOS drain conductance and the RRAM conductance at the access node,

$$\frac{\partial I}{\partial V_{\mathrm{BL}}} = \frac{g_{\mathrm{N},d}\, g_{\mathrm{R}}}{g_{\mathrm{N},d} + g_{\mathrm{R}}} \ge 0, \qquad \frac{\partial I}{\partial V_{\mathrm{SL}}} = \frac{g_{\mathrm{N},s}\, g_{\mathrm{R}}}{g_{\mathrm{N},d} + g_{\mathrm{R}}} \le 0,$$

where $g_{\mathrm{R}} = \partial I_{\mathrm{R}}/\partial(V_{\mathrm{BL}} - V_{\mathrm{X}}) \ge 0$ is the RRAM differential conductance and $g_{\mathrm{N},d} = \partial I_{\mathrm{N}}/\partial V_{\mathrm{X}} \ge 0$, $g_{\mathrm{N},s} = \partial I_{\mathrm{N}}/\partial V_{\mathrm{SL}} \le 0$ are the NMOS drain / source partials. The signs follow from the device I-V laws: the RRAM and NMOS-drain conductances are non-negative and the NMOS-source partial is non-positive, so $\partial I/\partial V_{\mathrm{BL}} \ge 0$ and $\partial I/\partial V_{\mathrm{SL}} \le 0$ for any operating point. The array wire Jacobian relies on these definite signs.

## Numerical method

The internal KCL $F_{\mathrm{X}}(V_{\mathrm{X}}) = 0$ is strictly monotone in $V_{\mathrm{X}}$: $I_{\mathrm{N}}$ is increasing in $V_{\mathrm{X}}$ and $I_{\mathrm{R}}$ is decreasing in $V_{\mathrm{X}}$, so $F_{\mathrm{X}} = I_{\mathrm{N}} - I_{\mathrm{R}}$ is strictly increasing and has a unique root for any $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$. The root is found by a Pade current-divider seed — a first-order split of the BL-to-SL drop across the NMOS output conductance and the programmed RRAM conductance — followed by a fixed number of Newton steps on $F_{\mathrm{X}}$. The step count is a calibrated numerical knob (see §Parameters); convergence is fast because the seed lands near the root and the residual decreases geometrically once in the Newton regime. The iteration count, unrolling, and compile behaviour are implementation choices — see [internals](../../../internals/xbar/_1t1r/cell.md).

## Noise & non-idealities

The cell owns no static mismatch of its own; non-idealities enter through its two device children, each gated by its own policy switch: RRAM conductance non-idealities (programming variation, drift, telegraph, thermal read noise) and access-NMOS threshold / transconductance mismatch — see [reference/device](../../device/README.md). The per-call snapshot fixes the read state, so the device noise is sampled once per call and the condensation is deterministic given that snapshot. The snapshot is taken at the per-call broadcast shape $(\dots, \text{col}, \text{row})$ with optional chunk selection, so each device sample aligns with the broadcast operating point the solver drives (the `shape` / `multi_coords` forwarding is in [internals](../../../internals/xbar/_1t1r/cell.md)).

## Parameters

The cell-owned parameters are the device configs, the access-NMOS sizing and parasitic-cap densities, the programming map, and the per-cell Newton count. They live in the cell config table `[xbar.core_config.cell_config]`.

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `rram_config` | RRAM storage-device configuration | — | see [reference/device/rram](../../device/rram.md) |
| `nmos_config` | access-NMOS configuration | — | see [reference/device/nmos](../../device/nmos.md) |
| `rram_g_max__uS` | maximum programmable RRAM conductance | uS | Measured |
| `state_to_g_map__uS` | state-index to target conductance (strictly increasing) | uS | Measured |
| `access_nmos_W__um` | access-NMOS width | um | Design |
| `access_nmos_L__um` | access-NMOS length | um | Design |
| `c_{gs,gd,db}_per_um__fF` | NMOS per-width parasitic-cap density | fF/um | Process |
| `n_newton` | per-cell access-node Newton iteration count | — | Calibrated (numerical convergence) |

Provenance terms are defined in [parameter_provenance](../../parameter_provenance.md). How to obtain values for a new chip: [calibration guide](../../../guides/calibration/README.md); file-level schema: [config reference](../../../api/README.md).

**Validation constraints.** Cross-field consistency is enforced at cell construction between the conductance map, the RRAM window, and the device floor $G_{\mathrm{min}}$ (`rram_config.g_min__uS`):

- $G_{\mathrm{RRAM,max}} > G_{\mathrm{min}}$, i.e. `rram_g_max__uS > rram_config.g_min__uS`;
- `state_to_g_map__uS` is strictly increasing and has length $\ge 2$;
- the first map entry stays at or above the device floor, `state_to_g_map__uS[0]` $\ge G_{\mathrm{min}}$;
- the last map entry stays at or below the programmable maximum, `state_to_g_map__uS[-1]` $\le G_{\mathrm{RRAM,max}}$.

## Energy model

The cell contributes the **device-capacitor dynamic energy** — the per-VMM charge/discharge energy of the capacitances internal to its own devices, summed at the converged operating point ($V_{\mathrm{BL}}$, $V_{\mathrm{SL}}$, and the condensed $V_{\mathrm{X}}$). A grounded cap dissipates $E = C\,V^2$ and a coupled cap $E = C\,(V_a - V_b)^2$:

- RRAM top electrode (BL side): grounded $C_{\mathrm{top}}\,V_{\mathrm{BL}}^2$;
- RRAM bottom electrode ($V_{\mathrm{X}}$ side): grounded $C_{\mathrm{bot}}\,V_{\mathrm{X}}^2$;
- NMOS drain-body: grounded $C_{db}\,V_{\mathrm{X}}^2$;
- NMOS gate-source: coupled $C_{gs}\,(V_{\mathrm{WL}} - V_{\mathrm{SL}})^2$;
- NMOS gate-drain: coupled $C_{gd}\,(V_{\mathrm{WL}} - V_{\mathrm{X}})^2$.

The wire-segment, control-line (WL), and DC-conduction energy is **not** the cell's — it is owned by [circuit_core](circuit_core.md). The cell carries no other PPA: its device children's silicon area and leakage roll up through the owning core's budget.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL}}$ | bit-line node voltage (cell terminal) | V | `v_bl` |
| $V_{\mathrm{SL}}$ | source-line node voltage (cell terminal) | V | `v_sl` |
| $V_{\mathrm{X}}$ | RRAM-NMOS internal access node | V | `XbarCell1T1RDCOP.v_x__V` |
| $V_{\mathrm{WL}}$ | word-line drive voltage (input) | V | `XbarCell1T1RSnapshot.v_wl__V` |
| $I$ | condensed branch current (BL $\to$ SL) | uA | `XbarCellDCOP.i__uA` |
| $I_{\mathrm{R}}$ | RRAM current | uA | `RRAM.solve_dc` |
| $I_{\mathrm{N}}$ | access-NMOS current | uA | `NMOS.solve_dc` |
| $g_{\mathrm{R}}$ | RRAM differential conductance | uS | `RRAM.solve_dc` |
| $g_{\mathrm{N},d}, g_{\mathrm{N},s}$ | NMOS drain / source partials | uS | `NMOS.solve_dc` |
| $\partial I/\partial V_{\mathrm{BL}}$ | BL-side branch conductance ($\ge 0$) | uS | `di_dvbl__uS` |
| $\partial I/\partial V_{\mathrm{SL}}$ | SL-side branch conductance ($\le 0$) | uS | `di_dvsl__uS` |
| $G_{\mathrm{RRAM,max}}$ | max programmable RRAM conductance | uS | `rram_g_max__uS` |
| $G_{\mathrm{min}}$ | RRAM device conductance floor | uS | `rram_config.g_min__uS` |
| $C_{\mathrm{top}}, C_{\mathrm{bot}}$ | RRAM electrode capacitances | fF | `rram.c_top__fF`, `rram.c_bot__fF` |
| $C_{gs}, C_{gd}, C_{db}$ | NMOS lumped parasitic capacitances | fF | `c_{gs,gd,db}_per_cell__fF` |

## Assumptions, scope & validity

Stated assumptions of the current model:

- The cell has exactly one internal node ($V_{\mathrm{X}}$); the series stack is RRAM then access NMOS.
- The solve is quasi-static: it finds the DC access-node operating point and does not model transient device switching within a pulse.
- The device-capacitor energy assumes a complete $0 \to \mathrm{DC} \to 0$ charge/discharge cycle per cap per WL pulse; the NMOS drain-body cap is referenced to $V_{\mathrm{X}}$.

TODO (domain author): the validity boundary of the lumped per-width NMOS parasitic-cap densities and any neglected internal-node capacitance.

## Validation

TODO: link the evidence in [validation/xbar](../../../validation/README.md) — cell branch-current and signed-conductance checks, internal-KCL residual, and finite-difference device-derivative checks.

## References

TODO: cite the RRAM and access-NMOS current models and the series-condensation basis.

---

- **Internals**: [cell internals](../../../internals/xbar/_1t1r/cell.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Configuration**: [config reference](../../../api/README.md) (`[xbar.core_config.cell_config]`)
- **Decisions**: [ADR-0003 pluggable xbar cell and the single nested solver](../../../about/adr/ADR-0003-xbar-cell-abstraction-and-single-nested-solver.md)
