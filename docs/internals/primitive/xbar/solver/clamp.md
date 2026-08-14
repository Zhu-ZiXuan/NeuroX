# Clamp driver role

A solve's boundary-clamp dependency is the structural `ClampDriver[SnapT, DcopT]` Protocol. It defines exactly one operation: `solve_dc(i_port__uA, snap, *, v_clamp_init__V) -> DcopT`, evaluated against a `ClampSnap` the caller has already sampled and returning the driver's own dcop under the `ClampDcop` bound. This is a capability role, not a registry base.

## Design decisions

- **The role is the transfer law alone.** Sampling a snap is not part of it: what a snapshot needs — which references to select, which shape the event covers, which axes are correlated across it — is knowledge the owning macro has and the solve does not, so `snapshot` is each concrete driver's own method, named and shaped by that driver, and a macro holds concrete classes to call it. The Protocol says what a boundary clamp *is* to a solve, and nothing about how a caller tiles its memory: a snap is sliced as an object after the fact, by the chunking layer alone.
- **A role, not a base class.** A clamp driver is a horizontal capability the solve needs from a boundary, not a place in a class tree. A shared base would force a second inheritance axis and a least-common-denominator config. The Protocol captures only the thin surface the solver consumes, so conformance is structural rather than nominal.
- **The result is a dcop the driver owns, bounded by what the solve reads.** `ClampDcop` requires two read-only properties — `v_clamp__V`, the port clamp voltage at the solved operating point, and `dvclamp_di__MOhm`, its derivative against the port current. Those two are the whole outer Newton's demand, so a concrete driver returns its own dcop dataclass carrying whatever else it computes, and the solve's per-call `DcopT` keeps that type unwidened. Returning a container rather than a tuple is what lets an iterative clamp publish its own convergence state without the role growing a position for it.
- **`ClampSnap.v_ref__V` is the NOMINAL reference.** The one field the role requires carries the ideal level and no driver-owned perturbation; an offset or a per-call draw lives in a field the concrete snap declares and its own `solve_dc` folds in. Two consumers depend on that split. The array reads `v_ref__V` as the ideal rest level its capacitive billing measures displacement from, which would otherwise charge a driver's non-ideality to the array's capacitance. The solver seeds its Newton from it, so the warm start is the nominal operating point: identical to the perturbed seed under an all-off policy, and with noise on it moves the starting point only, never the fixed point the iteration converges to.
- **Owned by the solver package.** Only the solve contract requires this surface, so the Protocol remains in `neurox/primitive/xbar/solver/` under the dependency-inversion rule that a role is defined with the operation that requires it.
- **Delivering the clamp is outside the role.** The role is what the solve consumes, and the solve consumes no delivery: a boundary is settled by `solve_dc` and billed afterwards by whoever owns the concrete block. A driver, a switched capacitor, a diode-connected load, and a transimpedance amplifier held in clamp all satisfy this role while each delivering through its own forward, under its own name and arguments; requiring one signature here would dictate an implementation shape instead of naming a capability. The owner knows which module it built, so the owner is what calls it.
- **A snap is what a driver samples, not a wrapper for a tensor.** `clamp.py` carries the clamp vocabulary alone: a `ClampSnap` exists because a driver has fabricated state to freeze before the solve — a nominal reference voltage, and whatever else that driver's own transfer law reads. A boundary the solve does not read, such as the word-line drive, has no such state and travels as the bare tensor it is, sliced by the same generic rule the chunking layer applies to every tensor operand.

## Gotchas

- **Do not narrow `ClampDriver`'s `SnapT` / `DcopT` bounds below `ClampSnap` / `ClampDcop`.** A concrete snap or dcop dataclass would turn structural conformance into an implementation dependency.
- **Do not re-bind the solver to a concrete clamp.** Replacing the per-call `ClampDriver[SnapT, DcopT]` parameters with concrete stored fields would defeat the role boundary.
- **`dvclamp_di__MOhm` is derivative-defined.** The role asks for ∂V_clamp/∂I, not for an output resistance a driver happens to hold: a clamp whose slope is not a stored constant computes it per call rather than reporting a config field.

---

- **Reference**: N/A — a structural software role; the clamp transfer characteristic is specified in [voltage driver](../../../../reference/primitive/analog/voltage_driver.md)
- **Implementation**: `neurox/primitive/xbar/solver/clamp.py`
- **Tests**: TODO — no dedicated clamp-role test
