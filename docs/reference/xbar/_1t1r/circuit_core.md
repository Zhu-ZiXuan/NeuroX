# 1T1R Cell Array (circuit_core)

## Summary

`CircuitCore1T1R` is the physical 1T1R cell array: it encapsulates and solves the physical array only, and is the **shared base** that the operating xbars ([offset](offset.md), and a future differential xbar) build on by adding a readout chain. It carries no encoding or readout itself. Each array site is one [cell](cell.md) sub-module — a two-terminal element between a bit-line node and a source-line node whose internal device topology the cell condenses. A VMM read drives the word lines, lets the array settle to a DC operating point under the interconnect parasitics and the boundary clamp-drivers, and exposes the per-column boundary result. This document specifies the array-level physical model, the operating-point equations, and the array energy model; the per-cell branch model is in [cell](cell.md), and the numerical method that solves the array is in [solver](../solver.md).

## Physical model

A cell at row $k$, column $c$ presents a two-terminal branch between its bit-line node $V_{\mathrm{BL},k}$ and source-line node $V_{\mathrm{SL},k}$, gated by the word-line voltage $V_{\mathrm{WL},k}$. The branch current and its two signed terminal conductances come from the [cell](cell.md), which condenses its own internal node; the array model below treats each cell as that condensed element and does not see the internal node. Per column two boundary clamp-drivers close the circuit: the BL clamp voltage $V_{\mathrm{BL,CL}}$ held by a transimpedance amplifier ([TIA](../../analog/README.md), the BL clamp-driver) absorbing the column's BL port current, and the SL clamp voltage $V_{\mathrm{SL,CL}}$ from the SL [driver](../../analog/README.md) (the SL clamp-driver). The word line is an input: $V_{\mathrm{WL},k}$ is produced by the WL [DAC](../../analog/README.md) from the integer activation code and carried into the cell snap.

Each line is an RC ladder. BL and SL are column-shared (one line per column), described per segment by a first driver-to-cell segment and a repeated cell-to-cell segment; IR drop develops along the interconnect-resistance segments, and which lines carry that resistance is design-dependent and resolved by the [solver](../solver.md) rather than pinned to a row or column axis here. WL is row-shared, carries no DC conduction path, and is treated as a single lumped capacitance along the row. The WL lumped capacitance per row is the first driver-to-cell segment plus the repeated cell-to-cell segments across the physical columns,

$$C_{\mathrm{WL,row}} = C_{\mathrm{WL,first}} + (N_{\mathrm{col}} - 1)\,C_{\mathrm{WL,seg}},$$

i.e. `c_wl_wire_per_row__fF = wl_first_c + (phys_col_num - 1) * wl_segment_c`.

Row and column are defined by **function**, not by a wiring scheme: a row shares one input; a column aggregates into one output.

## Governing equations

The DC operating point of one column over $N_{\mathrm{row}}$ rows has unknowns $\{V_{\mathrm{BL},k}, V_{\mathrm{SL},k}\}_{k=0}^{N_{\mathrm{row}}-1}$ plus the boundary scalars $V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}}$. The internal cell node is not an array unknown — each cell condenses it and reports a single branch current $I_{\mathrm{cell},k}(V_{\mathrm{BL},k}, V_{\mathrm{SL},k})$ (positive from $V_{\mathrm{BL}}$ to $V_{\mathrm{SL}}$) with its two signed terminal conductances, per [cell](cell.md). The residuals are the per-node wire-ladder KCL on the BL and SL lines,

$$F_{\mathrm{BL},k} = \operatorname{wire}_{\mathrm{BL},k}\!\left(V_{\mathrm{BL}}, V_{\mathrm{BL,CL}}\right) + I_{\mathrm{cell},k}\!\left(V_{\mathrm{BL},k}, V_{\mathrm{SL},k}\right) = 0,$$

$$F_{\mathrm{SL},k} = \operatorname{wire}_{\mathrm{SL},k}\!\left(V_{\mathrm{SL}}, V_{\mathrm{SL,CL}}\right) - I_{\mathrm{cell},k}\!\left(V_{\mathrm{BL},k}, V_{\mathrm{SL},k}\right) = 0,$$

and the two boundary constraints pinning the clamp voltages to their clamp-driver transfer functions at the port current,

$$F_{\mathrm{CL,BL}} = V_{\mathrm{BL,CL}} - \operatorname{TIA}\!\left(I_{\mathrm{BL,port}}\right) = 0, \qquad I_{\mathrm{BL,port}} = G_{\mathrm{seg},0}\,\left(V_{\mathrm{BL,CL}} - V_{\mathrm{BL},0}\right),$$

$$F_{\mathrm{CL,SL}} = V_{\mathrm{SL,CL}} - \operatorname{driver}_{\mathrm{SL}}\!\left(I_{\mathrm{SL,port}}\right) = 0, \qquad I_{\mathrm{SL,port}} = G_{\mathrm{seg},0}\,\left(V_{\mathrm{SL,CL}} - V_{\mathrm{SL},0}\right).$$

The same condensed branch current leaves the BL line ($F_{\mathrm{BL}}$ injects $I_{\mathrm{cell}}$) and enters the SL line ($F_{\mathrm{SL}}$ draws it), so the array sees one current per cell with no internal-node residual. The cell branch $I_{\mathrm{cell}}(\cdot)$ — set by the RRAM conductance $G_{\mathrm{RRAM}}$ in series with the access NMOS — is specified in [cell](cell.md); the boundary functions $\operatorname{TIA}(\cdot)$, $\operatorname{driver}_{\mathrm{SL}}(\cdot)$ in [reference/analog](../../analog/README.md). The system is solved by damped Newton iteration — see [solver](../solver.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL},k}$ | BL wire node voltage (row $k$) | V | `v_bl_node` |
| $V_{\mathrm{SL},k}$ | SL wire node voltage | V | `v_sl_node` |
| $V_{\mathrm{WL},k}$ | WL drive voltage (input) | V | `wl_dac.convert` output |
| $V_{\mathrm{BL,CL}}$ | BL clamp voltage (TIA) | V | `v_bl_clamp` |
| $V_{\mathrm{SL,CL}}$ | SL clamp voltage | V | `v_sl_drive` |
| $V_L, V_R$ | wire-segment endpoint voltages | V | adjacent node voltages |
| $V_a, V_b, V_{\mathrm{final}}$ | coupled / grounded cap node voltages | V | solver node voltages |
| $I_{\mathrm{cell},k}$ | condensed cell branch current (BL $\to$ SL) | uA | `cell.solve_branch` |
| $I_{\mathrm{BL,port}}, I_{\mathrm{SL,port}}$ | first-segment boundary port currents | uA | derived from node voltages |
| $G_{\mathrm{seg}}$ | wire segment conductance | uS | `bl_segment_g`, `sl_segment_g` |
| $G_{\mathrm{seg},0}$ | first wire-segment conductance | uS | `bl_segment_g[0]`, `sl_segment_g[0]` |
| $G_{\mathrm{RRAM}}$ | RRAM conductance | uS | `cell_config.state_to_g_map__uS` |
| $G_{\mathrm{RRAM,max}}$ | max programmable RRAM conductance | uS | `cell_config.rram_g_max__uS` |
| $G_{\mathrm{min}}$ | RRAM device conductance floor | uS | `cell_config.rram_config.g_min__uS` |
| $R_{\mathrm{seg}}$ | wire segment resistance | MOhm | `*_segment_r__MOhm` |
| $C$ | parasitic capacitance | fF | wire / NMOS cap fields |
| $E_{\mathrm{wire}}, E_{\mathrm{DC}}$ | per-VMM wire-cap / DC-conduction energy | fJ | `array_energy__fJ` |
| $t_{\mathrm{WL}}$ | WL pulse length | ns | `wl_pulse_length__ns` |
| $C_{\mathrm{WL,row}}$ | WL lumped capacitance per row | fF | `c_wl_wire_per_row__fF` |
| $C_{\mathrm{WL,first}}, C_{\mathrm{WL,seg}}$ | WL first / cell-to-cell segment cap | fF | `wl_first_c`, `wl_segment_c` |
| $N_{\mathrm{row}}$ | number of rows | — | `row_num` |
| $N_{\mathrm{col}}$ | number of physical columns | — | `phys_col_num` |

## Energy model

Per VMM the array dissipates wire-capacitor, control-line, DC-conduction, and device-capacitor energy. The **device-capacitor** term — the RRAM-electrode and access-NMOS capacitances internal to each cell — is owned by the [cell](cell.md) and summed over the array; the core owns the rest. The model assumes a full $0 \to \mathrm{DC} \to 0$ charge cycle per parasitic capacitor over one WL pulse. A grounded cap dissipates $E = C\,V_{\mathrm{final}}^2$ and a coupled cap $E = C\,(V_a - V_b)^2$ (no extra factor of two). BL/SL wire caps and the WL-line cap are the core's; BL/SL wire caps use a per-segment linear-voltage profile,

$$E_{\mathrm{wire}} = C\,\frac{V_L^2 + V_L V_R + V_R^2}{3},$$

where $V_L, V_R$ are the segment-endpoint voltages. DC conduction energy is the net supply power into the boundaries over the WL pulse, using the first-segment port currents $I_{\mathrm{BL,port}}$, $I_{\mathrm{SL,port}}$ defined above,

$$E_{\mathrm{DC}} = t_{\mathrm{WL}}\left(\sum_c V_{\mathrm{BL,CL}}\,I_{\mathrm{BL,port}} + \sum_c V_{\mathrm{SL,CL}}\,I_{\mathrm{SL,port}}\right).$$

By Tellegen's theorem $E_{\mathrm{DC}}$ equals the sum of the cell-branch and BL/SL wire-resistor Joule losses inside the array.

## Noise & non-idealities

The array owns no static mismatch; non-idealities enter through its children, each gated by a policy switch: per-cell RRAM conductance non-idealities and access-NMOS mismatch, carried inside the condensed branch — see [cell](cell.md); BL clamp-driver (TIA) finite gain and SL clamp-driver (driver) — see [reference/analog](../../analog/README.md); WL DAC drive noise — see [reference/analog](../../analog/README.md).

TODO: once the device/analog Reference documents exist, state exactly which sources couple into the operating point and how (e.g. how RRAM conductance variation perturbs the cell branch current $I_{\mathrm{cell}}$, how TIA finite gain shifts $V_{\mathrm{BL,CL}}$), with the statistical model per [notation_conventions](../../notation_conventions.md#noise-model-conventions).

## Parameters

The core owns a `cell` sub-module whose parameters (state map, RRAM window, access-NMOS sizing / parasitic caps, per-cell Newton count) live in the cell config table `[xbar.core_config.cell_config]` and are specified in [cell](cell.md); the core itself owns the interconnect ladder, WL pulse, and solver knobs.

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `cell_config` | 1T1R cell sub-module config (devices, sizing, state map, `n_newton`) | — | see [cell](cell.md) |
| BL/SL/WL `first_*`, `segment_*` $R_{\mathrm{seg}}$, $C$ | per-line interconnect ladder | MOhm, fF | Extracted |
| `wl_pulse_length__ns` | WL access duration (drives wire-RC charging energy) | ns | Design |
| solver iteration counts | numerical settling | — | Calibrated (numerical convergence) |

Provenance terms are defined in [parameter_provenance](../../parameter_provenance.md). How to obtain values for a new chip: [calibration guide](../../../guides/calibration/README.md); file-level schema: [config reference](../../../api/README.md). The cell's cross-field validation constraints (conductance map vs RRAM window vs device floor) are stated in [cell](cell.md).

## Assumptions, scope & validity

Stated assumptions of the current model:

- Interconnect is a lumped per-segment R/C ladder, not a distributed line.
- The WL line carries no DC conduction path and is a single lumped capacitance, uniform along the row.
- The energy model assumes a complete $0 \to \mathrm{DC} \to 0$ charge/discharge cycle per parasitic cap per WL pulse.
- The solve is quasi-static: it finds the DC operating point and does not model transient device switching within a pulse.

TODO (domain author): give the quantitative validity boundary — array-size range over which the lumped-segment approximation holds, the temperature treatment, neglected frequency-dependent / transient effects, and regimes where the model should not be trusted.

## Validation

TODO: link the evidence in [validation/xbar](../../../validation/README.md) — solver fixed-point and converged-residual checks, per-cell finite-difference device-derivative checks, chunking bit-exactness, and any array-vs-SPICE comparison.

## References

TODO: cite the RRAM and access-NMOS current models, the wire-ladder formulation, and the Tellegen energy-accounting basis.

---

- **Internals**: [circuit_core internals](../../../internals/xbar/_1t1r/circuit_core.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Configuration**: [config reference](../../../api/README.md) (`[xbar.core_config]`; cell devices / sizing / state map under `[xbar.core_config.cell_config]`)
- **Decisions**: [ADR-0004 clamp-driver role and the topology-agnostic array solver](../../../about/adr/ADR-0004-clamp-driver-protocol-and-generic-solver.md), [ADR-0003 pluggable xbar cell and the single nested solver](../../../about/adr/ADR-0003-xbar-cell-abstraction-and-single-nested-solver.md)
