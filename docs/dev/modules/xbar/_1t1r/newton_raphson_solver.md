# `neurox/xbar/_1t1r/newton_raphson_solver.py`

## Current role

`NewtonRaphsonSolver1T1R` solves the DC operating point of the fabricated 1T1R tile during one VMM call.

It is a **plain stateless tool class** — not an `nn.Module`, not a sub-module of `CircuitCore1T1R`'s module tree, and does not appear in `core.named_modules()` / `core.state_dict()`.

It holds long-lived references to the four boundary actors it must talk to (RRAM, NMOS, BL clamp driver, SL clamp driver) via `__init__`, because those are object identities that survive `.to(device)`. Everything else flows in per call.

## Ownership rule

The solver owns:

- no buffers
- no parameters
- no snapshots
- no wire tensors
- no runtime caches
- no `nn.Module` registration

The solver only references:

- the four boundary modules listed above (via `__init__`)
- class-level numerical constants (`N_UNROLL_OUTER`, `I_ATOL__uA`)

**Future revisions must not add instance state.** Wire conductances, snapshots, and any other per-call data must enter through `solve_dc` kwargs.

## `solve_dc` contract

Inputs:

- `v_wl_drive__V` — WL gate voltage per row, shape `[..., 1, num_row]`.
- `bl_segment_r__MOhm` / `sl_segment_r__MOhm` — 1-D wire segment resistances (driver-to-first at index 0).
- `bl_segment_g__uS` / `sl_segment_g__uS` — reciprocals of the above, supplied as buffers from the core so the solver does not allocate per call.
- `rram_snapshot` / `nmos_snapshot` — per-solve device snapshots sampled by the core.
- `bl_driver_snapshot` / `sl_driver_snapshot` — per-solve clamp-driver snapshots.

Output: `Solver1T1RDCOP` carrying the BL / SL / cell DC operating point.

## Boundary coupling

The solver is responsible for coupling the array interior to the BL / SL boundary conditions.

Important current rule:

- the BL clamp-driver contract is solver-facing and explicit
- the richer output voltage for readout is recovered after the outer solve by re-invoking the concrete BL clamp driver with the same snapshot

This keeps the solver's contract narrow while still allowing the readout path to reuse the full analog model.

## Runtime state

The solver works with per-call snapshots only. It does not mutate permanent device state during one solve, and holds no per-call data on `self`.
