# Crossbar array family

A crossbar pure array is a grid of [cell](../cell/family.md) sites bridged by resistive-capacitive interconnect, driven at the word lines and clamped at the bit-line and source-line boundaries. An array solve takes the analog word-line drive and two boundary clamp transfer functions, settles the grid to a DC operating point under the interconnect parasitics, and yields the per-column boundary port current and clamp voltage. The model is agnostic to the cell's internal device topology.

## Physical model

The array holds one cell at each column $c$ and row $k$. Every cell is a two-terminal branch between its bit-line node $V_{\mathrm{BL},k}$ and source-line node $V_{\mathrm{SL},k}$, gated by the word-line voltage $V_{\mathrm{WL},k}$; the branch current and its two signed terminal conductances come from the [cell](../cell/family.md), which condenses its own internal node so the array treats each site as a single condensed element and never sees the internal node. Each column's bit line and source line are resistive ladders along the row axis, one link per seat-to-seat step; IR drop develops along those links. Each circuit node carries its total capacitance to ground, including the device and interconnect parasitics seen there. The word line is the driven boundary, carries no DC conduction path, and enters the array as the input drive $V_{\mathrm{WL},k}$ at each seat's word-line node.

Two boundary clamp drivers close the circuit at each column: the bit-line clamp holds $V_{\mathrm{BL,CL}}$ while absorbing the column's bit-line port current, and the source-line driver holds $V_{\mathrm{SL,CL}}$. Both are peer blocks — the array does not own their transfer characteristics or their reference taps, but receives them per solve and pins its boundary voltages to them.

## Operating point

An array solve is defined over one column of $N_{\mathrm{row}}$ cells with unknowns the wire-node voltages $\{V_{\mathrm{BL},k}, V_{\mathrm{SL},k}\}$ plus the two boundary clamp scalars $V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}}$. The internal cell node is not an array unknown — each cell condenses it and reports a single branch current $I_{\mathrm{cell},k}(V_{\mathrm{BL},k}, V_{\mathrm{SL},k})$, positive from $V_{\mathrm{BL}}$ into $V_{\mathrm{SL}}$, with its two signed terminal conductances. The residuals are the per-node wire-ladder KCL on the two rails — the same condensed branch current leaves the bit-line KCL and enters the source-line KCL, so the array carries no per-cell internal residual — plus the two boundary constraints pinning the clamp voltages to the supplied transfer functions at the boundary port current. The solution yields the per-column bit-line port current $I_{\mathrm{BL,port}}$ and bit-line clamp voltage $V_{\mathrm{BL,CL}}$. The formulation, its well-posedness, and the block-tridiagonal linear algebra are specified in [solver](../solver/col_bl_col_sl.md); the per-cell condensed branch and its signed-conductance contract in [cell](../cell/family.md); the boundary transfer characteristic in [voltage driver](../../analog/voltage_driver.md).

## Programming

The array carries a conductance grid whose shape `(*inst_shape, phys_col, row)` fixes the mapped weight layout. A program step writes the cells from one state-index tensor whose shape matches that layout; each entry selects one cell's programmed device state. The state-to-conductance map, RRAM window, and device sizing are cell-level parameters, specified in [cell](../cell/family.md).

## Noise & non-idealities

The array owns no static mismatch of its own. Non-idealities enter through the cells — per-cell device conductance non-idealities and access-device mismatch, carried inside the condensed branch (see [cell](../cell/family.md)) — and through the injected boundary blocks, whose clamp-driver and word-line-drive non-idealities are properties of those peer blocks, not of the array.

TODO: once the device / analog Reference documents settle, state exactly which sources couple into the array operating point and how, with the statistical model per [nonideality](../../nonideality.md).

## Energy model

Per access the array dissipates the capacitive energy of its nodes under the supply-draw law in [capacitive energy](../../physics.md). Every node is billed at its own displacement and total capacitance to ground against the shared core analog supply. Duration-dependent DC-conduction energy and boundary-block energy are outside the array model. Array static PPA includes the cell-grid silicon area and leakage.

## Parameters

The array parameters are the repeated cell seat — its layout pitch, rail link resistance, and one capacitance total per node — plus the solver iteration counts. The model has no conduction-window parameter. Cell parameters are specified in [cell](../cell/family.md).

| Parameter | Meaning | Unit | Constraint | Source |
| --- | --- | --- | --- | --- |
| cell sub-module config | cell devices, sizing, state map, per-cell Newton count | — | — | see [cell](../cell/family.md) |
| cell-seat pitch | layout spacing between adjacent cell seats along each axis | um | $> 0$ | Extracted |
| bit-line / source-line link resistance | seat-to-seat rail interconnect | MOhm | $> 0$ | Extracted |
| per-node capacitance total | total capacitance to ground seen at each node of a seat | fF | $\ge 0$ | Extracted |
| solver iteration counts | numerical settling | — | integer $\ge 1$ | Calibrated (numerical convergence) |

Provenance terms are defined in [module_parameter](../../../../conventions/module_parameter.md). How to obtain values for a new chip: [calibration guide](../../../../guides/calibration/README.md); file-level schema: [config reference](../../../../api/README.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
| $V_{\mathrm{BL},k}$ | bit-line node voltage at row $k$ | V | `v_bl_node__V` |
| $V_{\mathrm{SL},k}$ | source-line node voltage | V | `v_sl_node__V` |
| $V_{\mathrm{WL},k}$ | word-line analog drive voltage (input) | V | `v_wl__V` |
| $V_{\mathrm{BL,CL}}$ | bit-line clamp voltage | V | `v_bl_clamp__V` |
| $V_{\mathrm{SL,CL}}$ | source-line clamp voltage | V | `v_sl_drive__V` |
| $I_{\mathrm{cell},k}$ | condensed cell branch current (BL $\to$ SL) | uA | `ResistiveDcop.i__uA` |
| $I_{\mathrm{BL,port}}, I_{\mathrm{SL,port}}$ | boundary port currents | uA | derived from node voltages |
| $N_{\mathrm{row}}$ | number of rows along each BL/SL rail ladder | — | `row_num` |
| $N_{\mathrm{col}}$ | number of physical columns | — | `col_num` |

## Assumptions, scope & validity

- Each array uses one topology-compatible cell family; within the BL/SL equations, every site contributes one condensed two-terminal branch and no internal cell node.
- Each column's BL/SL rails are lumped per-link resistive ladders along the row axis, not distributed lines.
- The word line is the driven boundary, carries no DC conduction path, and enters as the input drive at each seat's word-line node.
- Boundary transfer functions are supplied to each solve rather than modeled as part of the array.
- The solve is quasi-static: it finds the DC operating point and does not model transient device switching within a pulse.

TODO (domain author): the array-geometry and array-size range over which the lumped per-link / per-node abstraction holds, and regimes where a concrete array should not be trusted.

## Validation

TODO: add validation evidence for solver fixed points, converged residuals, per-cell finite-difference device derivatives, and chunking bit-exactness.

## References

TODO: cite the wire-ladder formulation and the Tellegen energy-accounting basis.
