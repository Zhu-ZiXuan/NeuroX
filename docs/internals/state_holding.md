# State Holding — Implementation

## Summary

Every fabricated circuit or device carries its physical state through three explicit stages — nominal (design intent), actual (post-fabrication / post-programming), snap (per-call dynamic noise) — each with its own representation, write moment, and name form. This document covers the state-ownership and snap contracts that hold across the module tree; the call lifecycle that drives the stage transitions is in [fabrication_lifecycle](fabrication_lifecycle.md).

## Design decisions

- **Three stages, not two.** Splitting actual from snap keeps the solver iterating against a frozen electrical characteristic: actual state (static mismatch) is resamplable but stable for the lifetime of a fabricated module, whereas snap state (kT/C, RTN, comparator decision noise) is the value the circuit sees at one instant of one DC solve. Folding dynamic noise into the actual buffer would let the characteristic drift between Newton iterations of the same call; folding mismatch into the snap would re-pay the sampling cost every call and lose the resample-once semantics. Rejected — a single mutable "current value" buffer (no clean resample boundary, aliasing across calls); chosen — three named tiers with distinct write moments.
- **The snap is a frozen dataclass, never a buffer.** A `@dataclass(frozen=True)` returned per call cannot be reassigned mid-solve and is discarded with the call, so no per-call state leaks onto the module and `state_dict` stays clean. Registering it as a buffer would persist call-local noise and invite the aliasing the three-stage split exists to prevent.
- **Owner-holds-state, parents do not mirror.** Fabricated state lives in the module that physically owns it (RRAM owns programmed conductance, NMOS owns `beta__uA_per_V2` / `vth__V`, OpAmpTIA owns its gain buffer); a parent reads child state through the child object. Mirroring child buffers on the parent would duplicate the source of truth and force the parent to know how to fabricate and interpret state it does not own. This boundary is what lets a future functional `program/fabricate + run` split reuse the same ownership lines without redesigning each child.

## Contracts & invariants

- **Canonical snapshot signature.** `snapshot(*, shape: tuple[int, ...]) -> <Name>Snap`, where `shape` is the per-call broadcast shape (distinct from `inst_shape`, the construction-time fabrication shape). Fields are `Tensor` or nested `*Snap` only — no plain Python objects — so the whole structure is tensor-migratable and consumable by `solve_dc(...)` / `convert(...)` / `vec_mat_mul(...)`.
- **Write moments are fixed per stage.** Nominal is written once (a circuit's `__init__`; the first step of an RRAM `program(...)`); actual is written by `_sample_fabricate_mismatch` (mismatch) or `program(...)` (programmed weight); snap is produced by `snapshot(*, shape)`. The actual value re-samples cleanly on every `fabricate()` — no accumulation across calls — using the per-instance shape committed to `self._inst_shape` at `__init__`.
- **Naming binds stage to symbol.** Nominal: scalar `<name>__<unit>` or buffer `nominal_<name>__<unit>`; actual: same name without the `nominal_` prefix; snap: fields of the `*Snap` dataclass. See [naming_conventions](../contributing/naming_conventions.md) and the symbol table in [notation_conventions](../reference/notation_conventions.md).
- **Nominal representation varies by owner.** A circuit's nominal is a scalar or 0-d buffer derived from `config` at `__init__`; an RRAM's nominal is the per-state conductance lookup (a Tensor indexed by programmed state, seeded in the first step of `program(...)`). Both feed the same `nominal_*` / actual naming, but the RRAM nominal is data-shaped, not a single config datum.
- **Cross-module snap passing is allowed and is not a new convention.** A solver may iterate on snaps owned by other modules (`NestedParallelRailSolver.solve_dc` reads the cell snap carrying `RRAMSnap` / `NMOSSnap`, and two clamp-driver snaps). The snap type carries its own semantics; the receiver's signature documents which type it expects. No extra rule beyond the snap contract above applies.
- **Per-call runtime state never reattaches to the module.** Anything created inside `solve_dc(...)` / `convert(...)` / `vec_mat_mul(...)` other than the snap is held in locals, returned via the `*DCOP` / `*Output` / output tensor, or emitted on the profiler side channel — never reassigned to a module attribute.

## Performance & resources

- The snap is allocated per call at the per-call broadcast `shape`, so dynamic-noise memory scales with the call's leading batch, not with `inst_shape`. It is freed when the call returns; nothing in the three-stage model retains per-call tensors. Resample cost (the actual stage) is paid only on `fabricate()`, and sampling reads the unchanged nominal buffers, so repeated fabricate calls do not grow memory.

## Gotchas

- **Do not treat the snap as durable state.** It is valid for exactly one DC solve. Caching a snap across calls reintroduces the mid-iteration drift the frozen-per-call rule forbids, and stale noise will silently bias a later solve.
- **`shape` (snap) is not `inst_shape` (fabrication).** The snap `shape` is the per-call broadcast shape; `inst_shape` is the construction-time per-instance multiplicity used to resample the actual value. Passing one where the other is expected mis-sizes the noise tensors.
- **A parent reading a duplicated child buffer is a bug.** If a parent appears to hold a child's fabricated value, it has violated the ownership rule; read through the child object instead.

## Known limitations

- N/A — the cross-cutting state model has no implementation TODOs of its own; per-subsystem state coverage is tracked with each subsystem's tests.

---

- **Reference**: N/A — cross-cutting software
- **Implementation**: `neurox/common/circuit.py`, `neurox/common/mixin/fabricate.py`
- **Tests**: `tests/test_signal_chain.py`, `tests/test_xbar_physics.py`
- **Decisions**: [ADR-0001](../about/adr/ADR-0001-config-dispatch-and-owned-construction.md), [ADR-0002](../about/adr/ADR-0002-nmos-is-a-pure-electrical-primitive.md)
