# 1T1R array

The 1T1R array joins a grid of [cell](../../cell/_1t1r/cell.md) sites with resistive-capacitive interconnect, driven at the word lines and clamped at the bit-line and source-line boundaries. Each site is a two-terminal element between a bit-line node and a source-line node whose internal device topology the cell condenses. An array solve takes the analog word-line drive and the two boundary clamp voltages, settles the array to a DC operating point under the interconnect parasitics, and yields the per-column boundary current and clamp voltage.

## Physical model

A cell at column $c$, row $k$ presents a two-terminal branch between its bit-line node $V_{\mathrm{BL},k}$ and source-line node $V_{\mathrm{SL},k}$, gated by the word-line voltage $V_{\mathrm{WL},k}$. The branch current and its two signed terminal conductances come from the [cell](../../cell/_1t1r/cell.md), which condenses its own internal node; the array model treats each cell as that condensed element and does not see the internal node. Two boundary clamp drivers close the circuit: the BL clamp voltage $V_{\mathrm{BL,CL}}$ held by the BL clamp driver absorbing the column's BL port current, and the SL clamp voltage $V_{\mathrm{SL,CL}}$ from the SL driver. The word line is an input: $V_{\mathrm{WL},k}$ is the analog word-line drive.

Each column's BL/SL rails are RC ladders along the row axis, described per segment by a first driver-to-cell segment and repeated cell-to-cell segments. The word line is the driven boundary, carries no DC conduction path, and is treated as a single lumped capacitance across the columns. The WL lumped capacitance is the first driver-to-cell segment plus the repeated cell-to-cell segments across the physical columns,

$$C_{\mathrm{WL,row}} = C_{\mathrm{WL,first}} + (N_{\mathrm{col}} - 1)\,C_{\mathrm{WL,seg}}.$$

## Governing equations

The DC operating point of one column over $N_{\mathrm{row}}$ cells has unknowns $\{V_{\mathrm{BL},k}, V_{\mathrm{SL},k}\}_{k=0}^{N_{\mathrm{row}}-1}$ plus the boundary scalars $V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}}$. The internal cell node is not an array unknown — each cell condenses it and reports a single branch current $I_{\mathrm{cell},k}(V_{\mathrm{BL},k}, V_{\mathrm{SL},k})$ (positive from $V_{\mathrm{BL}}$ to $V_{\mathrm{SL}}$) with its two signed terminal conductances, per [cell](../../cell/_1t1r/cell.md). The residuals are the per-node wire-ladder KCL on the BL and SL rails,

$$F_{\mathrm{BL},k} = \operatorname{wire}_{\mathrm{BL},k}\!\left(V_{\mathrm{BL}}, V_{\mathrm{BL,CL}}\right) + I_{\mathrm{cell},k}\!\left(V_{\mathrm{BL},k}, V_{\mathrm{SL},k}\right) = 0,$$

$$F_{\mathrm{SL},k} = \operatorname{wire}_{\mathrm{SL},k}\!\left(V_{\mathrm{SL}}, V_{\mathrm{SL,CL}}\right) - I_{\mathrm{cell},k}\!\left(V_{\mathrm{BL},k}, V_{\mathrm{SL},k}\right) = 0,$$

and the two boundary constraints pinning the clamp voltages to the clamp drivers' transfer functions at the port current,

$$F_{\mathrm{CL,BL}} = V_{\mathrm{BL,CL}} - \operatorname{driver}_{\mathrm{BL}}\!\left(I_{\mathrm{BL,port}}\right) = 0, \qquad I_{\mathrm{BL,port}} = G_{\mathrm{seg},0}\,\left(V_{\mathrm{BL,CL}} - V_{\mathrm{BL},0}\right),$$

$$F_{\mathrm{CL,SL}} = V_{\mathrm{SL,CL}} - \operatorname{driver}_{\mathrm{SL}}\!\left(I_{\mathrm{SL,port}}\right) = 0, \qquad I_{\mathrm{SL,port}} = G_{\mathrm{seg},0}\,\left(V_{\mathrm{SL,CL}} - V_{\mathrm{SL},0}\right).$$

The same condensed branch current leaves the BL rail ($F_{\mathrm{BL}}$ injects $I_{\mathrm{cell}}$) and enters the SL rail ($F_{\mathrm{SL}}$ draws it), so the array sees one current per cell with no internal-node residual. The cell branch $I_{\mathrm{cell}}(\cdot)$ — set by the RRAM conductance $G_{\mathrm{RRAM}}$ in series with the access NMOS — is specified in [cell](../../cell/_1t1r/cell.md); the boundary functions $\operatorname{driver}_{\mathrm{BL}}(\cdot)$, $\operatorname{driver}_{\mathrm{SL}}(\cdot)$ follow the [voltage-driver](../../../analog/voltage_driver.md) transfer characteristic. The operating-point solution yields the per-column BL port current $I_{\mathrm{BL,port}}$ and BL clamp voltage $V_{\mathrm{BL,CL}}$.

## Numerical method

The array operating point is solved by damped Newton iteration: an outer Newton on the clamp pair $(V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}})$ wraps an inner coupled block-$2\times2$ wire Newton on $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ along the row axis, with each cell condensing its internal node at every step. The formulation, its well-posedness, and the block-tridiagonal linear algebra are specified in [solver](../../solver/nested.md).

## Noise & non-idealities

The array owns no static mismatch of its own; non-idealities enter through the cells: per-cell RRAM conductance non-idealities and access-NMOS mismatch, carried inside the condensed branch — see [cell](../../cell/_1t1r/cell.md). The boundary clamp-driver and word-line-drive non-idealities are properties of those blocks, not of the array.

TODO: once the device/analog Reference documents exist, state exactly which sources couple into the operating point and how (e.g. how RRAM conductance variation perturbs the cell branch current $I_{\mathrm{cell}}$), with the statistical model per [nonideality](../../../nonideality.md).

## Parameters

The array's own parameters are the interconnect ladder and the solver iteration counts; the DC-conduction window is a per-solve argument, not a stored parameter. The cell sub-module's parameters (state map, RRAM window, access-NMOS sizing / parasitic caps, per-cell Newton count) live in the cell config table `[cim_macro.array_config.cell_config]`, specified in [cell](../../cell/_1t1r/cell.md).

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| `cell_config` | 1T1R cell sub-module config (devices, sizing, state map, `n_newton`) | — | — | see [cell](../../cell/_1t1r/cell.md) |
| BL/SL/WL `first_*` / `segment_*` R, C | array interconnect ladders | MOhm, fF | $> 0$ | Extracted |
| solver iteration counts | numerical settling | — | integer $\ge 1$ | Calibrated (numerical convergence) |

Provenance terms are defined in [module_parameter](../../../../../conventions/module_parameter.md). How to obtain values for a new chip: [calibration guide](../../../../../guides/calibration/README.md); file-level schema: [config reference](../../../../../api/README.md). The cell's cross-field validation constraints (conductance map vs RRAM window vs device floor) are stated in [cell detail](../../cell/_1t1r/cell_detail.md).

## Energy model

Per VMM the array dissipates wire-capacitor, control-line, DC-conduction, and per-cell node-capacitance energy. The **node-capacitance** term — the cell's four grounded node-to-ground capacitances — is a per-cell contribution ([cell](../../cell/_1t1r/cell.md)) summed over the array; the wire-capacitor, control-line, and DC-conduction terms are array-level. The model assumes a full $0 \to \mathrm{DC} \to 0$ charge cycle per capacitor per settled plane; a grounded cap dissipates $E = C\,V_{\mathrm{final}}^2$ (no extra factor of two). The array-level terms are the BL/SL wire caps and the WL-line cap; the BL/SL wire-cap energy uses a per-segment linear-voltage profile,

$$E_{\mathrm{wire}} = C\,\frac{V_L^2 + V_L V_R + V_R^2}{3},$$

where $V_L, V_R$ are the segment-endpoint voltages. DC conduction energy is the net supply power into the boundaries over the conduction window, using the first-segment port currents $I_{\mathrm{BL,port}}$, $I_{\mathrm{SL,port}}$ defined above,

$$E_{\mathrm{DC}} = t_{\mathrm{cond}}\left(\sum_c V_{\mathrm{BL,CL}}\,I_{\mathrm{BL,port}} + \sum_c V_{\mathrm{SL,CL}}\,I_{\mathrm{SL,port}}\right).$$

The conduction window $t_{\mathrm{cond}}$ is supplied per solve, not a stored array parameter: a scalar for a single settled plane, or a per-plane vector broadcasting against the solve leading when several input planes settle through one broadcast solve, scaling the DC-conduction energy plane by plane. By Tellegen's theorem $E_{\mathrm{DC}}$ equals the sum of the cell-branch and BL/SL wire-resistor Joule losses inside the array.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL},k}$ | BL wire node voltage at row $k$ | V | `v_bl_node` |
| $V_{\mathrm{SL},k}$ | SL wire node voltage | V | `v_sl_node` |
| $V_{\mathrm{WL},k}$ | WL analog drive voltage (input) | V | `v_wl` |
| $V_{\mathrm{BL,CL}}$ | BL clamp voltage | V | `v_bl_clamp` |
| $V_{\mathrm{SL,CL}}$ | SL clamp voltage | V | `v_sl_drive` |
| $V_L, V_R$ | wire-segment endpoint voltages | V | adjacent node voltages |
| $V_{\mathrm{final}}$ | grounded cap node voltage | V | solver node voltages |
| $I_{\mathrm{cell},k}$ | condensed cell branch current (BL $\to$ SL) | uA | `cell.solve_branch` |
| $I_{\mathrm{BL,port}}, I_{\mathrm{SL,port}}$ | first-segment boundary port currents | uA | derived from node voltages |
| $G_{\mathrm{seg}}$ | wire segment conductance | uS | `bl_segment_g__uS`, `sl_segment_g__uS` |
| $G_{\mathrm{seg},0}$ | first wire-segment conductance | uS | `bl_segment_g__uS[0]`, `sl_segment_g__uS[0]` |
| $G_{\mathrm{RRAM}}$ | RRAM conductance | uS | `cell_config.state_to_g_map__uS` |
| $G_{\mathrm{RRAM,max}}$ | max programmable RRAM conductance | uS | `cell_config.rram_g_max__uS` |
| $G_{\mathrm{min}}$ | RRAM device conductance floor | uS | `cell_config.rram_config.g_min__uS` |
| $R_{\mathrm{seg}}$ | wire segment resistance | MOhm | `*_segment_r__MOhm` |
| $C$ | parasitic capacitance | fF | wire / cell node-cap fields |
| $E_{\mathrm{wire}}, E_{\mathrm{DC}}$ | per-VMM wire-cap / DC-conduction energy | fJ | `array_energy__fJ` |
| $t_{\mathrm{cond}}$ | DC-conduction window (per-solve; scalar or per-plane) | ns | `t_conduct__ns` |
| $C_{\mathrm{WL,row}}$ | WL lumped capacitance per row | fF | `c_wl_wire_per_row__fF` |
| $C_{\mathrm{WL,first}}, C_{\mathrm{WL,seg}}$ | WL first / cell-to-cell segment cap | fF | `wl_first_c__fF`, `wl_segment_c__fF` |
| $N_{\mathrm{row}}$ | number of rows along each BL/SL wire ladder | — | `row_num` |
| $N_{\mathrm{col}}$ | number of physical columns | — | `col_num` |

## Assumptions, scope & validity

- Interconnect is a lumped per-segment R/C ladder, not a distributed line.
- The WL line carries no DC conduction path and is a single lumped capacitance, uniform across the columns.
- The energy model assumes a complete $0 \to \mathrm{DC} \to 0$ charge/discharge cycle per parasitic cap per settled plane.
- The solve is quasi-static: it finds the DC operating point and does not model transient device switching within a pulse.

TODO (domain author): give the quantitative validity boundary — array-size range over which the lumped-segment approximation holds, the temperature treatment, neglected frequency-dependent / transient effects, and regimes where the model should not be trusted.

## Validation

TODO: add validation evidence for solver fixed points, converged residuals, per-cell finite-difference device derivatives, chunking bit-exactness, and array-vs-SPICE comparison.

## References

TODO: cite the RRAM and access-NMOS current models, the wire-ladder formulation, and the Tellegen energy-accounting basis.

---

- **Internals**: [array internals](../../../../../internals/primitive/xbar/array/_1t1r/array.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Configuration**: [config reference](../../../../../api/README.md) (`[cim_macro.array_config]`; cell devices / sizing / state map under `[cim_macro.array_config.cell_config]`)
