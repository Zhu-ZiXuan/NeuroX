# 1T1R DC Solver

## Summary

The DC operating point of the [circuit_core](circuit_core.md) array is found by damped Newton iteration. The solver drives only the wire ladders and the clamp boundaries; each array site is a pluggable [cell](cell.md) that condenses its own internal device topology and presents a two-terminal branch. This document specifies the solving formulation, why the problem is well-posed, and the cell-pluggable contract the formulation rests on. Iteration counts, per-iteration linear algebra, memory cost, and compile behaviour are implementation choices, not physics — see [internals](../../../internals/xbar/_1t1r/solver.md).

## Formulation

The solve is a nested (block-Gauss-Seidel) decomposition: an *outer* $2\times2$ Newton on the per-column clamp pair $(V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}})$ wrapped around an *inner* array solve at a frozen clamp pair. The inner solve is a coupled block-$2\times2$ wire Newton on $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ along each line. The cell does not enter the array unknowns: at each inner step every cell condenses its internal node and returns one branch current with its two signed terminal conductances, which the wire Newton consumes directly.

## Cell-pluggable contract

The formulation is generic over the cell. At each array node the solver calls the cell's condensed branch solve and assembles the wire Jacobian from what it returns — it never models a device or an internal cell node itself. The contract the solver relies on:

- **Single branch current.** The cell returns one current $I_{\mathrm{cell}}(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ per site (positive from $V_{\mathrm{BL}}$ to $V_{\mathrm{SL}}$). The same current leaves the BL wire KCL and enters the SL wire KCL, so the array carries no per-cell internal residual.
- **Signed terminal derivatives.** The cell returns $\partial I_{\mathrm{cell}}/\partial V_{\mathrm{BL}} \ge 0$ and $\partial I_{\mathrm{cell}}/\partial V_{\mathrm{SL}} \le 0$. These definite signs are what make the inner wire system a well-posed M-matrix-flavour problem (below).
- **Snapshot-carried control.** Per-call device read state and the cell's control-line drive (the word line) travel in one cell snapshot; the solver passes that snapshot through and holds no cell-internal state.

Because the cell condenses any internal node, the solver is identical for any cell that satisfies this contract — the 1T1R series stack, or a future cell with a different internal topology. The per-cell condensation and its monotonicity are specified in [cell](cell.md).

## Well-posedness

The monotonicity directions follow from the cell's signed-conductance contract and the wire-ladder structure. The cell branch current is increasing in $V_{\mathrm{BL}}$ and decreasing in $V_{\mathrm{SL}}$ ($\partial I_{\mathrm{cell}}/\partial V_{\mathrm{BL}} \ge 0$, $\partial I_{\mathrm{cell}}/\partial V_{\mathrm{SL}} \le 0$), so the inner coupled wire system is a block-$2\times2$ tridiagonal M-matrix-flavour system with a unique fixed point at any frozen clamp pair. Because the cell branch currents are monotone in the node voltages, the boundary port currents $I_{\mathrm{BL,port}}$, $I_{\mathrm{SL,port}}$ are themselves monotone in the clamp voltages, so each boundary clamp-driver response is strictly monotone in a definite direction: raising $V_{\mathrm{BL,CL}}$ increases the cell read current and hence the BL port current it must absorb, while raising $V_{\mathrm{SL,CL}}$ lowers the cell drive and hence the SL port current. The outer map on $(V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}})$ composes the strictly monotone driver responses ($\operatorname{TIA}$ strictly monotone in $I_{\mathrm{BL,port}}$, SL driver strictly monotone in $I_{\mathrm{SL,port}}$) with the strictly monotone array response, giving a unique fixed point; the damped $2\times2$ Newton converges quadratically near it. Rail pseudo-equilibria are excluded, because the outer Newton is a well-conditioned per-column $2\times2$ problem away from the rails; a rail is reached only when the port current is genuinely outside the driver's reachable range, where the rail is the correct physics.

The solver carries IR drop through the per-segment interconnect resistances of the wire ladder: node voltages along each conducting line differ from the clamp voltage by the resistive drop the segment currents develop, and these drops enter the wire-ladder KCL residuals directly. Which lines carry the resistance is design-dependent (consistent with [circuit_core](circuit_core.md)); the solver makes no row-versus-column commitment and resolves the operating point for whatever segment-resistance ladder the configuration supplies.

## Symbols

As defined in [circuit_core](circuit_core.md#symbols) — the solver works on the same node voltages, boundary scalars, and condensed cell branch current.

## Validation

TODO: link [validation/xbar](../../../validation/README.md) — solver fixed-point and converged-residual checks, and per-cell finite-difference device-derivative checks.

## References

TODO: cite the Newton / block-tridiagonal solution methods.

---

- **Internals**: [solver internals](../../../internals/xbar/_1t1r/solver.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Decisions**: [ADR-0003 pluggable xbar cell and the single nested solver](../../../about/adr/ADR-0003-xbar-cell-abstraction-and-single-nested-solver.md)
