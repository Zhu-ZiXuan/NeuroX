# ADR-0004: Clamp-Driver Role and the Topology-Agnostic Array Solver

## Status

Accepted

## Context

[ADR-0003](ADR-0003-xbar-cell-abstraction-and-single-nested-solver.md) made the cell a pluggable, two-terminal abstraction and removed the global-Jacobian solver, leaving the nested block-Gauss-Seidel solve as the single 1T1R DC formulation. It condensed the *cell* but stopped at the cell: the solver still lived in a per-topology package (`neurox/xbar/_1t1r/`) and bound its two boundary actors as concrete circuit classes (`bl_driver: TIA`, `sl_driver: Driver`). Two consequences followed.

- **The boundary clamps were typed as kinds, not as a role.** A `TIA` (the BL clamp, closing a feedback loop on a virtual ground) and an ideal `Driver` (the SL clamp, a constant-voltage source) are physically unlike `CircuitBase` kinds that share no config, yet the solver depended on each concretely. The solver in fact uses only three capabilities of either boundary: a reference clamp voltage `v_ref__V`, a per-call `snapshot` of fabricated state, and a `solve_clamp` mapping port current to $(V_{\mathrm{clamp}}, \partial V_{\mathrm{clamp}}/\partial I)$. A future current-sense amplifier (CSA) boundary would offer the same three and nothing the solver needs differently, but could not be substituted without editing the solver's type signatures.

- **The solver was a 1T1R artifact, not a wire-network solver.** Despite consuming the cell only through the two-terminal `solve_branch` contract (ADR-0003), the solver's package, names, and DCOP type (`Solver1T1R`, `Solver1T1RDCOP`) were anchored to the 1T1R topology. The numerical method — IR-drop on coupled SL/BL wire ladders with a clamp at each boundary and a condensed branch at each crosspoint — has nothing 1T1R-specific in it once the cell is two-terminal. The per-topology framing blocked reuse for any future array topology and left ADR-0003's cell-agnostic intent only half-realized: the cell was swappable, but the solver that consumed it was not portable.

## Decision

### (a) The boundary clamp is a structural role (`Protocol`), not a registry-dispatched circuit kind

The clamp driver is a `ClampDriver` *role* — a structural `typing.Protocol[SnapT]` (`neurox/xbar/solver/clamp.py`) naming exactly the three members the array solve needs: the `v_ref__V` property, `snapshot(*, shape, multi_coords) -> SnapT`, and `solve_clamp(i_port__uA, snap, *, v_clamp_init__V) -> (v_clamp__V, dVclamp_dI__MOhm)`.

- **A role, not a type.** `TIA` and `Driver` are independent `CircuitBase` kinds that already share this surface by shape; they satisfy `ClampDriver` structurally with zero edits — no inheritance, no `RegistryMixin`. A future CSA conforms the same way, by matching the surface.
- **Generic over an unbounded `SnapT`.** `Protocol[SnapT]` ties a conforming clamp's `snapshot` output to its own `solve_clamp` input, so the snap type stays consistent end to end (`OpAmpTIA` is `ClampDriver[OpAmpTIASnap]`, `Driver` is `ClampDriver[DriverSnap]`). `SnapT` carries no bound on purpose: conforming snaps are unrelated dataclasses and need not share a base. A bound to any one clamp's snap would silently exclude the others.

**Alternative rejected — a `Clamp` abstract base with `RegistryMixin`.** Modeling the clamp as a config-keyed circuit *kind* (a `Clamp(CircuitBase, RegistryMixin)` family that `TIA`/`Driver` subclass) was rejected: it conflates *role* (the capability the solver consumes) with *kind* (what the circuit physically is), forces a least-common-denominator config across physically unlike clamps, stacks a second registry layer over the families' existing dispatch, and pushes `TIA`/`Driver` into multiple inheritance. The role is thin and structural; a base class is the wrong tool for it. A clamp's richer concrete handle (a TIA's own `solve_dc`) stays on the concrete class and is simply not named by the role.

### (b) The array solver is generalized to a topology-agnostic SL/BL IR-drop solver

The solver moves to its own package, `neurox/xbar/solver/`, as a topology-agnostic coupled SL/BL wire-IR-drop solver. The cell and the two clamp drivers become symmetric, swappable *actors of the solve call*, not members of the solver.

- **Actors are per-call, method-generic arguments.** `solve_dc` takes `cell`, `cell_snap`, `bl_driver`, `bl_driver_snap`, `sl_driver`, `sl_driver_snap` as arguments, generic over the cell's snap/DCOP types and the two driver snap types via method-level TypeVars. The class itself is non-generic. The cell is consumed through ADR-0003's two-terminal `solve_branch`; each driver through the `ClampDriver` role's three members. The solver names no concrete cell or clamp class.
- **The solver is stateless — config only.** `Solver` is `RegistryMixin[type[SolverConfig], Solver]`, built by `from_config(*, config=...)`, holding only its numerical configuration. It owns no cell, no driver, and no run state; every actor arrives at the call. The DCOP is `SolverDCOP(Generic[CellDCOPT])` so the returned cell operating point stays the caller's concrete cell DCOP type with no `Any`.
- **Driver-snap TypeVars are unbounded; the cell-snap TypeVar is bound.** The two driver snap TypeVars are unbounded, matching the `ClampDriver` role. The cell-snap TypeVar is bounded to `XbarCellSnap` because `XbarCell`'s own snap TypeVar is so bounded; an unbounded cell-snap would violate `XbarCell`'s contract.

This completes ADR-0003's intent. ADR-0003 made the *cell* a two-terminal abstraction the solver consumes through one contract; (b) makes the *solver itself* topology-agnostic and the *clamps* a role, so the whole DC solve is now a wire network plus swappable condensed-branch and boundary-clamp actors — no 1T1R-specific code on the solve path.

**Alternatives noted.**

- **Per-topology solver family (the previous design).** A `Solver1T1R` bound to concrete `TIA`/`Driver`/`XbarCell1T1R`. Rejected: the numerical method is topology-agnostic once the cell is two-terminal, so a per-topology class duplicates the same wire solve per topology and re-anchors the cell-agnostic intent to 1T1R.
- **A class-generic solver** (`Solver[CellSnapT, CellDCOPT, BLSnapT, SLSnapT]`, actors bound in `__init__`). Rejected as heavier: it puts four type parameters on the class, makes the solver carry per-actor state, and gains nothing — the actors already vary per call, so method generics express the binding without a stateful, four-parameter class.
- **Drivers in `__init__` typed `ClampDriver[Any]`.** Rejected: erasing the snap type to `Any` reintroduces exactly the `Any` leak the generic role exists to prevent and breaks the snapshot$\to$solve_clamp type link the `Protocol[SnapT]` enforces.

## Consequences

Positive:

- The DC solve is reusable across array topologies: a new topology supplies a two-terminal cell and two `ClampDriver` actors, with no solver change.
- New boundary clamps (a CSA) plug in by matching the role's three-member surface — no base class, no registry entry, no solver edit.
- The actor type flow is `Any`-free end to end: `from_config(config=...) -> solve_dc(cell, bl_driver, sl_driver, ...) -> SolverDCOP[<concrete cell DCOP>]`, with the cell DCOP, the cell's `solve_branch` result, and each driver's `solve_clamp` result all concrete.
- A stateless solver holds no per-call state, matching the rest of NeuroX's owned-construction discipline ([ADR-0001](ADR-0001-config-dispatch-and-owned-construction.md)).

Tradeoff / open item:

- **The clamp role is named but not yet uniformly consumed.** The nested solver's `solve_dc` accepts `ClampDriver[...]` actors, but the `_1t1r` glue that calls it still binds the two boundaries as concrete types (`bl_driver` narrowed to `OpAmpTIA`, `sl_driver` a `Driver`). The role is the seam; rewiring the remaining concrete bindings to the role is incremental, not a precondition.
- **Compile-verification gate (open).** The cell and clamp drivers are now `@torch.compile` *parameters* of the model's solve path rather than module attributes. Passing distinct actor instances as call arguments must preserve single-graph sharing — i.e. it must not induce per-instance recompilation or a graph break that fans out the compiled solver leaf. This is validated by the main session against the recompile / single-graph checks; the abstraction is accepted on the expectation that the gate passes, and the solve path stays self-compiled only at the numerical leaf with the model compiled by the caller.
