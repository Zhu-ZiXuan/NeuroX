# 1T1R core array (Core1T1R)

## Summary

`Core1T1R` is the pure physical 1T1R cell array: it encapsulates and solves the array only and is the **shared infrastructure** that a scheme xbar builds on by hoisting drivers, a boundary reference, and a readout chain above it. The core holds the cell array, the wire parasitics, and the generic solver; it owns no drivers, no DAC, no reference source, and no readout. Each array site is one [cell](cell.md) sub-module — a two-terminal element between a bit-line node and a source-line node whose internal device topology the cell condenses. An array solve takes the analog word-line drive and two boundary clamp drivers supplied by the scheme xbar, lets the array settle to a DC operating point under the interconnect parasitics, and returns the per-line boundary result. This document specifies the array-level physical model, the operating-point equations, and the array energy model; the per-cell branch model is in [cell](cell.md), and the numerical method that solves the array is in [solver](../solver.md).

## Physical model

A cell at series position $k$, parallel line $c$ presents a two-terminal branch between its bit-line node $V_{\mathrm{BL},k}$ and source-line node $V_{\mathrm{SL},k}$, gated by the word-line voltage $V_{\mathrm{WL},k}$. The branch current and its two signed terminal conductances come from the [cell](cell.md), which condenses its own internal node; the array model below treats each cell as that condensed element and does not see the internal node. Two boundary clamp drivers — supplied by the scheme xbar, not owned by the core — close the circuit: the BL clamp voltage $V_{\mathrm{BL,CL}}$ held by the BL clamp driver absorbing the line's BL port current, and the SL clamp voltage $V_{\mathrm{SL,CL}}$ from the SL driver. The word line is an input: $V_{\mathrm{WL},k}$ is the analog drive handed in by the xbar (which already ran its WL DAC) and carried into the cell snap.

Each line is an RC ladder, described per segment by a first driver-to-cell segment and a repeated cell-to-cell segment; IR drop develops along the interconnect-resistance segments, and which lines carry that resistance is resolved by the [solver](../solver.md) rather than pinned to an axis here. The word line is the driven boundary, carries no DC conduction path, and is treated as a single lumped capacitance along the series. The WL lumped capacitance is the first driver-to-cell segment plus the repeated cell-to-cell segments across the physical parallel lines,

$$C_{\mathrm{WL,row}} = C_{\mathrm{WL,first}} + (N_{\mathrm{line}} - 1)\,C_{\mathrm{WL,seg}},$$

i.e. `c_wl_wire_per_row__fF = wl_first_c + (phys_col_num - 1) * wl_segment_c`.

## Governing equations

The DC operating point of one BL/SL line over $N_{\mathrm{series}}$ cells has unknowns $\{V_{\mathrm{BL},k}, V_{\mathrm{SL},k}\}_{k=0}^{N_{\mathrm{series}}-1}$ plus the boundary scalars $V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}}$. The internal cell node is not an array unknown — each cell condenses it and reports a single branch current $I_{\mathrm{cell},k}(V_{\mathrm{BL},k}, V_{\mathrm{SL},k})$ (positive from $V_{\mathrm{BL}}$ to $V_{\mathrm{SL}}$) with its two signed terminal conductances, per [cell](cell.md). The residuals are the per-node wire-ladder KCL on the BL and SL lines,

$$F_{\mathrm{BL},k} = \operatorname{wire}_{\mathrm{BL},k}\!\left(V_{\mathrm{BL}}, V_{\mathrm{BL,CL}}\right) + I_{\mathrm{cell},k}\!\left(V_{\mathrm{BL},k}, V_{\mathrm{SL},k}\right) = 0,$$

$$F_{\mathrm{SL},k} = \operatorname{wire}_{\mathrm{SL},k}\!\left(V_{\mathrm{SL}}, V_{\mathrm{SL,CL}}\right) - I_{\mathrm{cell},k}\!\left(V_{\mathrm{BL},k}, V_{\mathrm{SL},k}\right) = 0,$$

and the two boundary constraints pinning the clamp voltages to the clamp drivers' transfer functions at the port current,

$$F_{\mathrm{CL,BL}} = V_{\mathrm{BL,CL}} - \operatorname{driver}_{\mathrm{BL}}\!\left(I_{\mathrm{BL,port}}\right) = 0, \qquad I_{\mathrm{BL,port}} = G_{\mathrm{seg},0}\,\left(V_{\mathrm{BL,CL}} - V_{\mathrm{BL},0}\right),$$

$$F_{\mathrm{CL,SL}} = V_{\mathrm{SL,CL}} - \operatorname{driver}_{\mathrm{SL}}\!\left(I_{\mathrm{SL,port}}\right) = 0, \qquad I_{\mathrm{SL,port}} = G_{\mathrm{seg},0}\,\left(V_{\mathrm{SL,CL}} - V_{\mathrm{SL},0}\right).$$

The same condensed branch current leaves the BL line ($F_{\mathrm{BL}}$ injects $I_{\mathrm{cell}}$) and enters the SL line ($F_{\mathrm{SL}}$ draws it), so the array sees one current per cell with no internal-node residual. The cell branch $I_{\mathrm{cell}}(\cdot)$ — set by the RRAM conductance $G_{\mathrm{RRAM}}$ in series with the access NMOS — is specified in [cell](cell.md); the boundary functions $\operatorname{driver}_{\mathrm{BL}}(\cdot)$, $\operatorname{driver}_{\mathrm{SL}}(\cdot)$ are the clamp drivers passed in by the scheme xbar, in [reference/analog](../../analog/README.md). The system is solved by damped Newton iteration — see [solver](../solver.md). The core returns the reassembled boundary result — the per-line BL port current $I_{\mathrm{BL,port}}$ and BL clamp voltage $V_{\mathrm{BL,CL}}$ — which the scheme xbar reads out.

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL},k}$ | BL wire node voltage (series $k$) | V | `v_bl_node` |
| $V_{\mathrm{SL},k}$ | SL wire node voltage | V | `v_sl_node` |
| $V_{\mathrm{WL},k}$ | WL analog drive voltage (input) | V | `v_wl` |
| $V_{\mathrm{BL,CL}}$ | BL clamp voltage | V | `v_bl_clamp` |
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
| $C_{\mathrm{WL,row}}$ | WL lumped capacitance per series | fF | `c_wl_wire_per_row__fF` |
| $C_{\mathrm{WL,first}}, C_{\mathrm{WL,seg}}$ | WL first / cell-to-cell segment cap | fF | `wl_first_c`, `wl_segment_c` |
| $N_{\mathrm{series}}$ | number of cells along a line (series axis) | — | `row_num` |
| $N_{\mathrm{line}}$ | number of physical parallel lines | — | `phys_col_num` |

## Energy model

Per VMM the array dissipates wire-capacitor, control-line, DC-conduction, and device-capacitor energy. The **device-capacitor** term — the RRAM-electrode and access-NMOS capacitances internal to each cell — is owned by the [cell](cell.md) and summed over the array; the core owns the rest. The model assumes a full $0 \to \mathrm{DC} \to 0$ charge cycle per parasitic capacitor over one WL pulse. A grounded cap dissipates $E = C\,V_{\mathrm{final}}^2$ and a coupled cap $E = C\,(V_a - V_b)^2$ (no extra factor of two). BL/SL wire caps and the WL-line cap are the core's; BL/SL wire caps use a per-segment linear-voltage profile,

$$E_{\mathrm{wire}} = C\,\frac{V_L^2 + V_L V_R + V_R^2}{3},$$

where $V_L, V_R$ are the segment-endpoint voltages. DC conduction energy is the net supply power into the boundaries over the WL pulse, using the first-segment port currents $I_{\mathrm{BL,port}}$, $I_{\mathrm{SL,port}}$ defined above,

$$E_{\mathrm{DC}} = t_{\mathrm{WL}}\left(\sum_c V_{\mathrm{BL,CL}}\,I_{\mathrm{BL,port}} + \sum_c V_{\mathrm{SL,CL}}\,I_{\mathrm{SL,port}}\right).$$

By Tellegen's theorem $E_{\mathrm{DC}}$ equals the sum of the cell-branch and BL/SL wire-resistor Joule losses inside the array. The driver, DAC, and readout energies are not the core's — each scheme-xbar peer self-logs its own.

## Noise & non-idealities

The array owns no static mismatch; non-idealities enter through its children, each gated by a policy switch: per-cell RRAM conductance non-idealities and access-NMOS mismatch, carried inside the condensed branch — see [cell](cell.md). The boundary clamp-driver and WL DAC non-idealities belong to the scheme xbar that owns those peers, not to the core.

TODO: once the device/analog Reference documents exist, state exactly which sources couple into the operating point and how (e.g. how RRAM conductance variation perturbs the cell branch current $I_{\mathrm{cell}}$), with the statistical model per [nonideality](../../nonideality.md).

## Parameters

The core owns a `cell` sub-module whose parameters (state map, RRAM window, access-NMOS sizing / parasitic caps, per-cell Newton count) live in the cell config table `[xbar.core_config.cell_config]` and are specified in [cell](cell.md); the core itself owns the interconnect ladder, WL pulse, and solver knobs.

| Parameter | Meaning | Unit | Source |
|---|---|---|---|
| `cell_config` | 1T1R cell sub-module config (devices, sizing, state map, `n_newton`) | — | see [cell](cell.md) |
| BL/SL/WL `first_*`, `segment_*` $R_{\mathrm{seg}}$, $C$ | per-line interconnect ladder | MOhm, fF | Extracted |
| `wl_pulse_length__ns` | WL access duration (drives wire-RC charging energy) | ns | Design |
| solver iteration counts | numerical settling | — | Calibrated (numerical convergence) |

Provenance terms are defined in [module_parameter](../../../conventions/module_parameter.md). How to obtain values for a new chip: [calibration guide](../../../guides/calibration/README.md); file-level schema: [config reference](../../../api/README.md). The cell's cross-field validation constraints (conductance map vs RRAM window vs device floor) are stated in [cell](cell.md).

## Assumptions, scope & validity

Stated assumptions of the current model:

- Interconnect is a lumped per-segment R/C ladder, not a distributed line.
- The WL line carries no DC conduction path and is a single lumped capacitance, uniform along the series.
- The energy model assumes a complete $0 \to \mathrm{DC} \to 0$ charge/discharge cycle per parasitic cap per WL pulse.
- The solve is quasi-static: it finds the DC operating point and does not model transient device switching within a pulse.

TODO (domain author): give the quantitative validity boundary — array-size range over which the lumped-segment approximation holds, the temperature treatment, neglected frequency-dependent / transient effects, and regimes where the model should not be trusted.

## Validation

TODO: link the evidence in [validation/xbar](../../../validation/README.md) — solver fixed-point and converged-residual checks, per-cell finite-difference device-derivative checks, chunking bit-exactness, and any array-vs-SPICE comparison.

## References

TODO: cite the RRAM and access-NMOS current models, the wire-ladder formulation, and the Tellegen energy-accounting basis.

---

- **Internals**: [Core1T1R internals](../../../internals/xbar/_1t1r/core.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Configuration**: [config reference](../../../api/README.md) (`[xbar.core_config]`; cell devices / sizing / state map under `[xbar.core_config.cell_config]`)
- **Decisions**: [ADR-0004 clamp-driver role and the topology-agnostic array solver](../../../about/adr/ADR-0004-clamp-driver-protocol-and-generic-solver.md), [ADR-0003 pluggable xbar cell and the single nested solver](../../../about/adr/ADR-0003-xbar-cell-abstraction-and-single-nested-solver.md)
