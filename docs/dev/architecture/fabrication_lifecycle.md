# Fabrication Lifecycle

This document records the current object lifecycle rules.

## Split of responsibilities

NeuroX generally separates four phases:

1. **`__init__`** Bind config, design parameters, and stable runtime context such as `dtype` and `T__K`.
2. **`fabricate(...)` / `program(...)`** Bind shape-derived or programmed static physical state.
3. **`snapshot(...)`** Materialize per-call runtime state from the fabricated object.
4. **`forward(...)` / `solve_dc(...)` / `convert(...)`** Use fabricated state and runtime snapshots to perform one operation.

## Why this split exists

- shape-derived state should not be forced into construction
- fabricated state is usually longer-lived than one call
- runtime state should stay local to one call
- this split keeps a future functional `prepare/program + run` API possible

## Current rule

When a module needs shape or weight-programmed state, it should not guess that state at `__init__` time. Instead:

- construction binds stable identity and owned children;
- `fabricate(...)` binds shape / programmed state;
- `forward(...)` consumes the already-fabricated module.

## Canonical `fabricate` signature

Every leaf circuit (analog / digital / device) exposes the same canonical `fabricate` signature:

```python
def fabricate(self, shape: tuple[int, ...]) -> None: ...
```

- `shape` is the per-instance fabrication shape — always positional, always present, even when the body is a no-op.
- `fabricate` takes no other arguments. Structural facts (cap widths, per-bank cap counts, per-digit positional weights, …) are bound in `__init__` so the per-instance state sampling stays shape-only.
- Structural facts threaded through `__init__` must arrive as plain Python types (`int`, `tuple[float, ...]`, …); tensor construction happens once inside the leaf's own `__init__`.
- A leaf without per-instance state still accepts `shape` and ignores it; this keeps every caller (`xbar`, `readout`, future family-level orchestrators) calling the same lifecycle method.

Tile-level wrappers (`xbar`, `macro`) instead take a programmed weight (`fabricate(w)`) — they fan a single user-facing payload out across the leaf chain. The same shape-only rule applies to every leaf they call into.

## Re-callability and nominal templates

Every `fabricate(...)` call must be idempotent on its inputs:

- the leaf caches its nominal templates at construction (`nominal_*` buffers / scalars);
- `fabricate(...)` rebuilds the fabricated buffers from those nominals via `clone().expand(shape)`;
- mismatch / noise sources are gated by `cfg.enable_<source>` toggles consumed by the `apply_*` helpers (see [`noise_and_toggles.md`](noise_and_toggles.md)); disabling a source is a toggle change, never a `None`-substitution of a nominal value.

Re-calling `fabricate(...)` after a previous call therefore always restarts from the unchanged nominals — there is no path through which fabricated state accumulates across calls.
