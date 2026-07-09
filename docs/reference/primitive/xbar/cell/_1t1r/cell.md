# 1T1R cell

The 1T1R cell places one RRAM device in series with an access NMOS, $\mathrm{BL} - \mathrm{RRAM} - V_{\mathrm{X}} - \mathrm{NMOS} - \mathrm{SL}$, with the NMOS gate driven by the word line. At a terminal pair $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ and word-line drive $V_{\mathrm{WL}}$ it condenses its internal access node $V_{\mathrm{X}}$ and presents a single two-terminal branch — one branch current and two signed terminal conductances — following the [cell family](../README.md) conventions.

## Physical model

The cell has one internal node, the access node $V_{\mathrm{X}}$ between the RRAM and the access NMOS. The RRAM conducts between the bit line $V_{\mathrm{BL}}$ and $V_{\mathrm{X}}$; the access NMOS conducts between $V_{\mathrm{X}}$ and the source line $V_{\mathrm{SL}}$, gated by the word-line voltage $V_{\mathrm{WL}}$. The EKV access-NMOS model is source/drain symmetric, so source and drain labels are a naming convention: the SL-side terminal ($V_{\mathrm{SL}}$) is taken as the source and the internal BL-side terminal ($V_{\mathrm{X}}$) as the drain, with no effect on the device current. The word-line drive and the device read state are inputs fixed per call.

The cell presents a **two-terminal branch**: the internal node $V_{\mathrm{X}}$ is condensed away inside the cell, so the cell is one element between $V_{\mathrm{BL}}$ and $V_{\mathrm{SL}}$ whose current and terminal conductances summarize the entire series stack. The RRAM and NMOS device transfer functions $I_{\mathrm{R}}(\cdot)$ and $I_{\mathrm{N}}(\cdot)$ are specified in [reference/device](../../../device/README.md).

## Governing equations

At a fixed terminal pair $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ the access node satisfies the internal KCL — the RRAM current into $V_{\mathrm{X}}$ equals the NMOS current out of it,

$$F_{\mathrm{X}} = I_{\mathrm{N}}\!\left(V_{\mathrm{WL}}, V_{\mathrm{X}}, V_{\mathrm{SL}}\right) - I_{\mathrm{R}}\!\left(V_{\mathrm{BL}} - V_{\mathrm{X}}\right) = 0.$$

Solving this for $V_{\mathrm{X}}$ condenses the cell. The condensed branch then reports a **single consistent current**: at convergence $I_{\mathrm{R}} = I_{\mathrm{N}}$, and that common value is the branch current $I$ (positive from $V_{\mathrm{BL}}$ to $V_{\mathrm{SL}}$). The residual $\lvert I_{\mathrm{N}} - I_{\mathrm{R}} \rvert$ at the condensed $V_{\mathrm{X}}$ is the per-cell convergence diagnostic.

The cell also returns the two **signed terminal conductances** of its condensed branch, obtained by eliminating $V_{\mathrm{X}}$ from the series stack — the series condensation of the NMOS drain conductance and the RRAM conductance at the access node,

$$\frac{\partial I}{\partial V_{\mathrm{BL}}} = \frac{g_{\mathrm{N},d}\, g_{\mathrm{R}}}{g_{\mathrm{N},d} + g_{\mathrm{R}}} \ge 0, \qquad \frac{\partial I}{\partial V_{\mathrm{SL}}} = \frac{g_{\mathrm{N},s}\, g_{\mathrm{R}}}{g_{\mathrm{N},d} + g_{\mathrm{R}}} \le 0,$$

where $g_{\mathrm{R}} = \partial I_{\mathrm{R}}/\partial(V_{\mathrm{BL}} - V_{\mathrm{X}}) \ge 0$ is the RRAM differential conductance and $g_{\mathrm{N},d} = \partial I_{\mathrm{N}}/\partial V_{\mathrm{X}} \ge 0$, $g_{\mathrm{N},s} = \partial I_{\mathrm{N}}/\partial V_{\mathrm{SL}} \le 0$ are the NMOS drain / source partials. The signs follow from the device I-V laws: the RRAM and NMOS-drain conductances are non-negative and the NMOS-source partial is non-positive, so $\partial I/\partial V_{\mathrm{BL}} \ge 0$ and $\partial I/\partial V_{\mathrm{SL}} \le 0$ for any operating point.

## Numerical method

The internal KCL $F_{\mathrm{X}}(V_{\mathrm{X}}) = 0$ is strictly monotone in $V_{\mathrm{X}}$: $I_{\mathrm{N}}$ is increasing in $V_{\mathrm{X}}$ and $I_{\mathrm{R}}$ is decreasing in $V_{\mathrm{X}}$, so $F_{\mathrm{X}} = I_{\mathrm{N}} - I_{\mathrm{R}}$ is strictly increasing and has a unique root for any $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$. The root is found by a Pade current-divider seed — a first-order split of the BL-to-SL drop across the NMOS output conductance and the programmed RRAM conductance — followed by a fixed number of Newton steps on $F_{\mathrm{X}}$. The step count is a calibrated numerical knob (see §Parameters); convergence is fast because the seed lands near the root and the residual decreases geometrically once in the Newton regime.

## Noise & non-idealities

The cell introduces no static mismatch of its own; non-idealities enter through its two devices: RRAM conductance non-idealities (programming variation, drift, telegraph, thermal read noise) and access-NMOS threshold / transconductance mismatch — see [reference/device](../../../device/README.md). The read state is fixed per call, so the device noise is sampled once per call and the condensation is deterministic given that state.

## Parameters

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `rram_config` | RRAM storage-device configuration | — | — | see [reference/device/rram](../../../device/rram.md) |
| `nmos_config` | access-NMOS configuration | — | — | see [reference/device/mosfet](../../../device/mosfet.md) |
| `rram_g_max__uS` | maximum programmable RRAM conductance | uS | $> G_{\mathrm{min}}$ | Design |
| `state_to_g_map__uS` | state-index to target conductance | uS | strictly increasing; length $\ge 2$; within $[G_{\mathrm{min}}, G_{\mathrm{RRAM,max}}]$ | Calibrated (physical data) |
| `access_nmos_W__um` | access-NMOS width | um | $> 0$ | Design |
| `access_nmos_L__um` | access-NMOS length | um | $> 0$ | Design |
| `c_{gs,gd,db}_per_um__fF` | NMOS per-width parasitic-cap density | fF/um | $\ge 0$ | Process |
| `n_newton` | per-cell access-node Newton iteration count | — | $\ge 1$ | Calibrated (numerical convergence) |

Provenance terms are defined in [module_parameter](../../../../../conventions/module_parameter.md). How to obtain values for a new chip: [calibration guide](../../../../../guides/calibration/README.md); file-level schema: [config reference](../../../../../api/README.md).

## Energy model

The cell contributes the **device-capacitor dynamic energy** — the charge/discharge energy of the capacitances internal to its own devices, summed at the converged operating point ($V_{\mathrm{BL}}$, $V_{\mathrm{SL}}$, and the condensed $V_{\mathrm{X}}$). A grounded cap dissipates $E = C\,V^2$ and a coupled cap $E = C\,(V_a - V_b)^2$:

- RRAM top electrode (BL side): grounded $C_{\mathrm{top}}\,V_{\mathrm{BL}}^2$;
- RRAM bottom electrode ($V_{\mathrm{X}}$ side): grounded $C_{\mathrm{bot}}\,V_{\mathrm{X}}^2$;
- NMOS drain-body: grounded $C_{db}\,V_{\mathrm{X}}^2$;
- NMOS gate-source: coupled $C_{gs}\,(V_{\mathrm{WL}} - V_{\mathrm{SL}})^2$;
- NMOS gate-drain: coupled $C_{gd}\,(V_{\mathrm{WL}} - V_{\mathrm{X}})^2$.

Wire-segment, control-line (WL), and DC-conduction energy, together with the devices' silicon area and static leakage, lie outside the cell's energy model.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL}}$ | bit-line node voltage (cell terminal) | V | `v_bl` |
| $V_{\mathrm{SL}}$ | source-line node voltage (cell terminal) | V | `v_sl` |
| $V_{\mathrm{X}}$ | RRAM-NMOS internal access node | V | `XbarCell1T1RDCOP.v_x__V` |
| $V_{\mathrm{WL}}$ | word-line drive voltage (input) | V | `XbarCell1T1RSnap.v_wl__V` |
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

TODO: link the evidence in [validation/xbar](../../../../../validation/README.md) — cell branch-current and signed-conductance checks, internal-KCL residual, and finite-difference device-derivative checks.

## References

TODO: cite the RRAM and access-NMOS current models and the series-condensation basis.

---

- **Internals**: [cell internals](../../../../../internals/primitive/xbar/cell/_1t1r/cell.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Configuration**: [config reference](../../../../../api/README.md) (`[cim_macro.array_config.cell_config]`)
