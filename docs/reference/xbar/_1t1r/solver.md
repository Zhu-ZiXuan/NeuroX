# 1T1R DC Solver

## Summary

The DC operating point of the [circuit_core](circuit_core.md) array is found by damped Newton iteration. This document specifies the solving formulations and why the problem is well-posed. Iteration counts, per-iteration linear algebra, memory cost, and compile behaviour are implementation choices, not physics — see [internals](../../../internals/xbar/_1t1r/solver.md).

## Formulations

Two mathematically equivalent formulations are provided and cross-checked against each other:

- **Nested (block-Gauss-Seidel)** — decompose into an *inner* array solve at a frozen clamp pair and an *outer* $2\times2$ Newton on $(V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}})$ per column. The inner solve alternates a per-cell Newton for $V_{\mathrm{X}}$ with a coupled block-$2\times2$ wire Newton for $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$.
- **Full-Jacobian** — every unknown enters one global Newton step on a block-tridiagonal Jacobian, with the two boundary scalars Schur-eliminated against the first array row.

Both converge to the same operating point.

## Well-posedness

The monotonicity directions follow from the device I-V laws. The RRAM read current is a hyperbolic-sine function of the voltage across it, $I_{\mathrm R}(V_{\mathrm{BL}}-V_{\mathrm{X}})$ strictly increasing in $V_{\mathrm{BL}}-V_{\mathrm{X}}$ (so increasing in $V_{\mathrm{BL}}$, decreasing in $V_{\mathrm{X}}$); the access-NMOS current $I_{\mathrm N}(V_{\mathrm{WL}}, V_{\mathrm{X}}, V_{\mathrm{SL}})$ is strictly increasing in the applied drain-source voltage $V_{\mathrm{X}}-V_{\mathrm{SL}}$ (so increasing in $V_{\mathrm{X}}$, decreasing in $V_{\mathrm{SL}}$). With $I_{\mathrm N}$ increasing in $V_{\mathrm{X}}$ and $I_{\mathrm R}$ decreasing in $V_{\mathrm{X}}$, the cell residual $F_{\mathrm{X},k} = I_{\mathrm N} - I_{\mathrm R}$ is strictly increasing in $V_{\mathrm{X}}$, so each cell KCL has a unique $V_{\mathrm{X}}$ for any $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$, and the inner coupled wire system is a block-$2\times2$ tridiagonal M-matrix-flavour system with a unique fixed point at any frozen clamp pair. Because the per-cell currents are monotone in the node voltages, the boundary port currents $I_{\mathrm{BL,port}}$, $I_{\mathrm{SL,port}}$ are themselves monotone in the clamp voltages, so each boundary clamp-driver response is strictly monotone in a definite direction: raising $V_{\mathrm{BL,CL}}$ increases the RRAM read current and hence the BL port current it must absorb, while raising $V_{\mathrm{SL,CL}}$ lowers the NMOS drain-source drive and hence the SL port current. The outer map on $(V_{\mathrm{BL,CL}}, V_{\mathrm{SL,CL}})$ composes the strictly monotone driver responses ($\operatorname{TIA}$ strictly monotone in $I_{\mathrm{BL,port}}$, SL driver strictly monotone in $I_{\mathrm{SL,port}}$) with the strictly monotone array response, giving a unique fixed point; the damped $2\times2$ Newton converges quadratically near it. Rail pseudo-equilibria that can trap a single simultaneous Newton are excluded, because the outer Newton is a well-conditioned per-column $2\times2$ problem away from the rails; a rail is reached only when the port current is genuinely outside the driver's reachable range, where the rail is the correct physics.

The solver carries IR drop through the per-segment interconnect resistances of the wire ladder: node voltages along each conducting line differ from the clamp voltage by the resistive drop the segment currents develop, and these drops enter the wire-ladder KCL residuals directly. Which lines carry the resistance is design-dependent (consistent with [circuit_core](circuit_core.md)); the solver makes no row-versus-column commitment and resolves the operating point for whatever segment-resistance ladder the configuration supplies.

## Symbols

As defined in [circuit_core](circuit_core.md#symbols) — the solver works on the same node voltages, currents, and boundary scalars.

## Validation

TODO: link [validation/xbar](../../../validation/README.md) — solver fixed-point and finite-difference Jacobian checks, and nested-vs-full-Jacobian agreement.

## References

TODO: cite the Newton / block-tridiagonal solution methods.

---

- **Internals**: [solver internals](../../../internals/xbar/_1t1r/solver.md)
- **Validation**: TODO — `validation/xbar` (not yet written)
- **Decisions**: N/A — no ADR governs this module.
