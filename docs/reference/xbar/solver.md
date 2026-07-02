# Parallel BL/SL DC solver

## Summary

The DC operating point of a crossbar array with **parallel BL/SL rails** is found by damped Newton iteration. The solver drives only the two wire ladders and the two clamp boundaries; every array site is a pluggable [cell](cell.md) that condenses its own internal device topology and presents a two-terminal branch, and each line boundary is a swappable clamp driver. This document specifies the solving formulation, why the problem is well-posed, the four structural assumptions the formulation rests on, and the cell- and driver-pluggable contracts. The solver is cell- and driver-agnostic: it is identical for the 1T1R array and for any other parallel-rail topology whose cell condenses to one branch. Iteration counts, per-iteration linear algebra, memory cost, and compile behaviour are implementation choices, not physics.

## Structural assumptions

The formulation is specialised to a **parallel BL/SL** array — the BL rail and the SL rail run side by side along one shared series direction — and rests on four assumptions:

1. **Exactly two array rails.** Each site couples a BL node and an SL node; the coupled wire Newton is therefore a block-$2\times2$ per node.
2. **One signed two-terminal branch.** The two rails couple only through a single signed cell branch current; the cell self-condenses any internal node, so the array carries no per-cell internal unknown.
3. **The control line is a driven boundary.** The gate/control line (the word line) is an externally driven boundary, not a solved mesh node. Hence the parallel (per-driver) lines are independent and are batched along the parallel axis.
4. **Each rail is a 1-D series ladder.** IR drop accumulates along one series axis per rail, giving the (block-)tridiagonal structure the Thomas sweep exploits.

The orthogonal case (BL $\perp$ SL forming a 2-D mesh, where the two rails are *not* parallel and a line is a solved node) violates assumption 3 and is a **future, separate solver** — out of scope here.

## Canonical layout

The two array rails share one series axis. Of the two trailing axes of the cell grid $[\dots, \mathrm{line}, \mathrm{series}]$, the **last** axis is the series (wire-ladder / IR-drop) direction and the second-to-last is the parallel (per-driver) line axis. The solver body runs in this canonical orientation; a construction-time integer `series_axis` (default $-1$, supplied by the core, not a config field) names the caller's series axis, and the solver normalises a non-canonical caller (`series_axis = -2`) to canonical at entry and restores the caller's layout at exit. The solver makes no row-versus-column commitment beyond this single series-axis choice.

## Formulation

The solve is a nested (block-Gauss-Seidel) decomposition: an *outer* $2\times2$ Newton on the per-column clamp pair $(V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}})$ wrapped around an *inner* array solve at a frozen clamp pair. The inner solve is a coupled block-$2\times2$ wire Newton on $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ along each line. The cell does not enter the array unknowns: at each inner step every cell condenses its internal node and returns one branch current with its two signed terminal conductances, which the wire Newton consumes directly.

## Pluggable actor contracts

The formulation is generic over the cell and over both clamp drivers. The solver holds none of them as state; each is supplied per solve. At every array node the solver calls the cell's condensed branch solve and assembles the wire Jacobian from what it returns, and at each line boundary it calls the clamp driver's transfer function — it never models a device, an internal cell node, or a specific driver itself.

The cell contract:

- **Single branch current.** The cell returns one current $I_{\mathrm{cell}}(V_{\mathrm{BL}}, V_{\mathrm{SL}})$ per site (positive from $V_{\mathrm{BL}}$ to $V_{\mathrm{SL}}$). The same current leaves the BL wire KCL and enters the SL wire KCL, so the array carries no per-cell internal residual.
- **Signed terminal derivatives.** The cell returns $\partial I_{\mathrm{cell}}/\partial V_{\mathrm{BL}} \ge 0$ and $\partial I_{\mathrm{cell}}/\partial V_{\mathrm{SL}} \le 0$. These definite signs are what make the inner wire system a well-posed M-matrix-flavour problem (below).
- **Snap-carried control.** Per-call device read state and the cell's control-line drive (the word line) travel in one cell snap; the solver passes that snap through and holds no cell-internal state.

The clamp-driver contract (one driver per line boundary, BL and SL):

- **Monotone scalar transfer.** Each driver maps its boundary port current to a clamp voltage with a strict, definite-sign response: the BL clamp driver is strictly monotone in $I_{\mathrm{BL,port}}$, the SL driver strictly monotone in $I_{\mathrm{SL,port}}$. The outer Newton needs only this scalar response and its derivative.
- **Snap-carried boundary state.** Each driver's per-call state travels in its own driver snap; the solver passes both snaps through and holds no boundary state.

Because the cell condenses any internal node and both drivers reduce to a monotone scalar boundary law, the solver is identical for any cell and driver pair satisfying these contracts — the 1T1R series stack with an op-amp TIA on BL, or a different cell and a different boundary driver. The per-cell condensation and its monotonicity are specified in [cell](cell.md); the clamp-driver transfer functions in [reference/analog](../analog/README.md).

## Well-posedness

The monotonicity directions follow from the cell's signed-conductance contract and the wire-ladder structure. The cell branch current is increasing in $V_{\mathrm{BL}}$ and decreasing in $V_{\mathrm{SL}}$ ($\partial I_{\mathrm{cell}}/\partial V_{\mathrm{BL}} \ge 0$, $\partial I_{\mathrm{cell}}/\partial V_{\mathrm{SL}} \le 0$), so the inner coupled wire system is a block-$2\times2$ tridiagonal M-matrix-flavour system with a unique fixed point at any frozen clamp pair. Because the cell branch currents are monotone in the node voltages, the boundary port currents $I_{\mathrm{BL,port}}$, $I_{\mathrm{SL,port}}$ are themselves monotone in the clamp voltages, so each boundary clamp-driver response is strictly monotone in a definite direction: raising $V_{\mathrm{BL,CL}}$ increases the cell read current and hence the BL port current it must absorb, while raising $V_{\mathrm{SL,CL}}$ lowers the cell drive and hence the SL port current. The outer map on $(V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}})$ composes the strictly monotone driver responses (the BL clamp driver strictly monotone in $I_{\mathrm{BL,port}}$, the SL driver strictly monotone in $I_{\mathrm{SL,port}}$) with the strictly monotone array response, giving a unique fixed point; the damped $2\times2$ Newton converges quadratically near it. Rail pseudo-equilibria are excluded, because the outer Newton is a well-conditioned per-column $2\times2$ problem away from the rails; a rail is reached only when the port current is genuinely outside the driver's reachable range, where the rail is the correct physics.

The solver carries IR drop through the per-segment interconnect resistances of the wire ladder: node voltages along the series axis differ from the clamp voltage by the resistive drop the segment currents develop, and these drops enter the wire-ladder KCL residuals directly. The solver resolves the operating point for whatever per-segment resistance ladder the configuration supplies along the series axis.

## Symbols

Shared electrical symbols are pinned in [notation_conventions](../../conventions/notation_conventions.md); the per-call cell branch quantities are defined in [cell](cell.md).

| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
| $V_{\mathrm{BL}}$ | BL wire node voltage | V | `v_bl_node` |
| $V_{\mathrm{SL}}$ | SL wire node voltage | V | `v_sl_node` |
| $V_{\mathrm{BL,CL}}$ | BL clamp voltage | V | `v_bl_clamp` |
| $V_{\mathrm{SL,CL}}$ | SL clamp voltage | V | `v_sl_drive` |
| $I_{\mathrm{cell}}$ | condensed cell branch current (BL $\to$ SL) | uA | `cell.solve_branch` |
| $\partial I_{\mathrm{cell}}/\partial V_{\mathrm{BL}}$ | BL-side branch conductance ($\ge 0$) | uS | `cell.solve_branch` |
| $\partial I_{\mathrm{cell}}/\partial V_{\mathrm{SL}}$ | SL-side branch conductance ($\le 0$) | uS | `cell.solve_branch` |
| $I_{\mathrm{BL,port}}$ | BL boundary port current | uA | derived from node voltages |
| $I_{\mathrm{SL,port}}$ | SL boundary port current | uA | derived from node voltages |

## Validation

TODO: link [validation/xbar](../../validation/README.md) — solver fixed-point and converged-residual checks, and per-cell finite-difference device-derivative checks.

## References

TODO: cite the Newton / block-tridiagonal solution methods.

---

- **Internals**: [solver internals](../../internals/xbar/solver.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Configuration**: [config reference](../../api/README.md) (`[xbar.core_config.solver_config]`)
- **Decisions**: [ADR-0004 clamp-driver role and the topology-agnostic array solver](../../about/adr/ADR-0004-clamp-driver-protocol-and-generic-solver.md), [ADR-0003 pluggable xbar cell and the single nested solver](../../about/adr/ADR-0003-xbar-cell-abstraction-and-single-nested-solver.md)
