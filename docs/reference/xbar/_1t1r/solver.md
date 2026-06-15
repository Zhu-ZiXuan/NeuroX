# 1T1R DC Solver

## Summary

The DC operating point of the [circuit_core](circuit_core.md) array is found by damped Newton iteration. This document specifies the solving formulations and why the problem is well-posed. Iteration counts, per-iteration linear algebra, memory cost, and compile behaviour are implementation choices, not physics — see [internals](../../../internals/xbar/_1t1r/solver.md).

## Formulations

Two mathematically equivalent formulations are provided and cross-checked against each other:

- **Nested (block-Gauss-Seidel)** — decompose into an *inner* array solve at a frozen clamp pair and an *outer* $2\times2$ Newton on $(V_{\mathrm{BL,CL}}, V_{\mathrm{SL,DR}})$ per column. The inner solve alternates a per-cell Newton for $V_{\mathrm{X}}$ with a coupled block-$2\times2$ wire Newton for $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$.
- **Full-Jacobian** — every unknown enters one global Newton step on a block-tridiagonal Jacobian, with the two boundary scalars Schur-eliminated against the first array row.

Both converge to the same operating point.

## Well-posedness

With $I_{\mathrm N}(V_{\mathrm{X}})$ monotone against $I_{\mathrm R}(V_{\mathrm{BL}}-V_{\mathrm{X}})$, each cell KCL has a unique $V_{\mathrm{X}}$ for any $(V_{\mathrm{BL}}, V_{\mathrm{SL}})$, so the inner coupled wire system is a block-$2\times2$ tridiagonal M-matrix-flavour system with a unique fixed point at any frozen clamp pair. The outer map on $(V_{\mathrm{BL,CL}}, V_{\mathrm{SL,DR}})$ composes the monotone driver responses ($\operatorname{TIA}$ monotone in $I_{\mathrm{BL,port}}$, SL driver monotone in $I_{\mathrm{SL,port}}$) with the monotone array response, giving a unique fixed point; the damped $2\times2$ Newton converges quadratically near it. Rail pseudo-equilibria that can trap a single simultaneous Newton are excluded, because the outer Newton is a well-conditioned per-column $2\times2$ problem away from the rails; a rail is reached only when the port current is genuinely outside the driver's reachable range, where the rail is the correct physics.

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
