# `neurox/xbar/_1t1r/newton_raphson_solver.py`

## Current role

`NewtonRaphsonSolver1T1R` solves the DC operating point of the fabricated 1T1R tile during one VMM call.

It consumes:

- programmed / fabricated device snapshots
- boundary-driver snapshots
- drive voltages prepared by the core

It does not own long-lived physical state itself.

## Boundary coupling

The solver is responsible for coupling the array interior to the BL / SL boundary conditions.

Important current rule:

- the BL clamp-driver contract is solver-facing and explicit
- the richer output voltage for readout is recovered after the outer solve by re-invoking the concrete BL clamp driver with the same snapshot

This keeps the solver's contract narrow while still allowing the readout path to reuse the full analog model.

## Runtime state

The solver works with per-call snapshots only. It does not mutate permanent device state during one solve.
