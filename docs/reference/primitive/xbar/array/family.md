# Xbar array family

A crossbar pure array is the shared physical body of one compute-in-memory tile: a grid of [cell](../cell/family.md) sites bridged by resistive-capacitive interconnect, driven at the word lines and clamped at the bit-line and source-line boundaries. It owns only the cell grid, the wire parasitics, and the DC solver; the boundary blocks (word-line drive, bit-line clamp, source-line drive, boundary voltage reference) are peers under the scheme macro, injected into an array solve rather than owned. An array solve takes the analog word-line drive and the two boundary clamp references, settles the array to a DC operating point under the interconnect parasitics, and yields the per-column boundary port current and clamp voltage the macro readout consumes. This layer is agnostic to the cell's internal device topology; a concrete array realizes it for one array geometry.

## Physical model

The array holds one cell at each column $c$ and row $k$. Every cell is a two-terminal branch between its bit-line node $V_{\mathrm{BL},k}$ and source-line node $V_{\mathrm{SL},k}$, gated by the word-line voltage $V_{\mathrm{WL},k}$; the branch current and its two signed terminal conductances come from the [cell](../cell/family.md), which condenses its own internal node so the array treats each site as a single condensed element and never sees the internal node. Each column's bit line and source line are resistive-capacitive ladders along the row axis; IR drop develops along their resistive interconnect segments. The word line is the driven boundary, carries no DC conduction path, and enters the array only as the input drive $V_{\mathrm{WL},k}$ plus a lumped line capacitance.

Two boundary clamp drivers close the circuit at each column: the bit-line clamp holds $V_{\mathrm{BL,CL}}$ while absorbing the column's bit-line port current, and the source-line driver holds $V_{\mathrm{SL,CL}}$. Both are peer blocks — the array does not own their transfer characteristics or their reference taps, but receives them per solve and pins its boundary voltages to them.

## Operating point

An array solve is defined over one column of $N_{\mathrm{row}}$ cells with unknowns the wire-node voltages $\{V_{\mathrm{BL},k}, V_{\mathrm{SL},k}\}$ plus the two boundary clamp scalars $V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}}$. The internal cell node is not an array unknown — each cell condenses it and reports a single branch current $I_{\mathrm{cell},k}(V_{\mathrm{BL},k}, V_{\mathrm{SL},k})$, positive from $V_{\mathrm{BL}}$ into $V_{\mathrm{SL}}$, with its two signed terminal conductances. The residuals are the per-node wire-ladder KCL on the two rails — the same condensed branch current leaves the bit-line KCL and enters the source-line KCL, so the array carries no per-cell internal residual — plus the two boundary constraints pinning the clamp voltages to the injected clamp drivers' transfer functions at the boundary port current. The solution yields the per-column bit-line port current $I_{\mathrm{BL,port}}$ and bit-line clamp voltage $V_{\mathrm{BL,CL}}$, the two quantities the macro readout consumes. The formulation, its well-posedness, and the block-tridiagonal linear algebra are specified in [solver](../solver/nested.md); the per-cell condensed branch and its signed-conductance contract in [cell](../cell/family.md); the boundary clamp-driver transfer characteristic in [voltage driver](../../analog/voltage_driver.md).

## Programming

The array carries a conductance grid whose shape `(*inst, phys_col, row)` fixes the mapped weight layout. A program step writes the cells from one state-index tensor whose shape matches that layout; each entry selects one cell's programmed device state. The state-to-conductance map, RRAM window, and device sizing are cell-level parameters, specified in [cell](../cell/family.md).

## Noise & non-idealities

The array owns no static mismatch of its own. Non-idealities enter through the cells — per-cell device conductance non-idealities and access-device mismatch, carried inside the condensed branch (see [cell](../cell/family.md)) — and through the injected boundary blocks, whose clamp-driver and word-line-drive non-idealities are properties of those peer blocks, not of the array.

TODO: once the device / analog Reference documents settle, state exactly which sources couple into the array operating point and how, with the statistical model per [nonideality](../../nonideality.md).

## Energy model

Per VMM the array dissipates wire-capacitor, control-line, DC-conduction, and per-cell node-capacitance energy. The node-capacitance term — each cell's per-node grounded capacitances — is a per-cell contribution ([cell](../cell/family.md)) summed over the array; the wire-capacitor, control-line, and DC-conduction terms are array-level. The array is a full electrical circuit and aggregates the device children's silicon area and static leakage at the array level (the cells contribute only per-read node-capacitance switching energy). The injected boundary blocks self-account their own drive / reference / readout energy.

## Parameters

The array's own parameters are the interconnect ladder, the word-line pulse, and the solver iteration counts; the cell sub-module's parameters (state map, device window, sizing, per-cell Newton count) live in the cell config sub-tree, specified in [cell](../cell/family.md).

| Parameter | Meaning | Unit | Constraint | Source |
|---|---|---|---|---|
| cell sub-module config | cell devices, sizing, state map, per-cell Newton count | — | — | see [cell](../cell/family.md) |
| bit-line / source-line / word-line ladder R, C | array interconnect ladders | MOhm, fF | $> 0$ | Extracted |
| word-line pulse length | access duration (drives wire-RC charging energy) | ns | $> 0$ | Design |
| solver iteration counts | numerical settling | — | integer $\ge 1$ | Calibrated (numerical convergence) |

Provenance terms are defined in [module_parameter](../../../../conventions/module_parameter.md). How to obtain values for a new chip: [calibration guide](../../../../guides/calibration/README.md); file-level schema: [config reference](../../../../api/README.md).

## Symbols

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL},k}$ | bit-line wire node voltage at row $k$ | V | `v_bl_node` |
| $V_{\mathrm{SL},k}$ | source-line wire node voltage | V | `v_sl_node` |
| $V_{\mathrm{WL},k}$ | word-line analog drive voltage (input) | V | `v_wl` |
| $V_{\mathrm{BL,CL}}$ | bit-line clamp voltage | V | `v_bl_clamp` |
| $V_{\mathrm{SL,CL}}$ | source-line clamp voltage | V | `v_sl_drive` |
| $I_{\mathrm{cell},k}$ | condensed cell branch current (BL $\to$ SL) | uA | `cell.solve_branch` |
| $I_{\mathrm{BL,port}}, I_{\mathrm{SL,port}}$ | boundary port currents | uA | derived from node voltages |
| $N_{\mathrm{row}}$ | number of rows along each BL/SL wire ladder | — | `row_num` |
| $N_{\mathrm{col}}$ | number of physical columns | — | `col_num` |

## Assumptions, scope & validity

- The array is topology-agnostic in the cell: it sees each site only as one condensed two-terminal branch and holds no internal cell node.
- Each column's BL/SL rails are lumped per-segment R/C ladders along the row axis, not distributed lines.
- The word line is the driven boundary, carries no DC conduction path, and enters as the input drive plus a lumped line capacitance.
- The boundary blocks (drive, clamp, reference, readout) are peers injected per solve, not owned by the array.
- The solve is quasi-static: it finds the DC operating point and does not model transient device switching within a pulse.

TODO (domain author): the array-geometry and array-size range over which the lumped-segment abstraction holds, and regimes where a concrete array should not be trusted.

## Validation

TODO: add validation evidence for solver fixed points, converged residuals, per-cell finite-difference device derivatives, and chunking bit-exactness.

## References

TODO: cite the wire-ladder formulation and the Tellegen energy-accounting basis.

---

- **Internals**: [array base](../../../../internals/primitive/xbar/array/base.md)
- **Concrete array**: [1T1R core array](_1t1r/array.md)
- **Configuration**: [config reference](../../../../api/README.md)
