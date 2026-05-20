# ADR-0001: Config Dispatch and Owned Construction

## Status

Accepted

## Context

Older NeuroX code often threaded external factory closures through parent objects. That made ownership and construction logic hard to read:

- the parent appeared to own a child without constructing it
- design parameters for child devices leaked across layers
- static typing was weak
- historical docstrings drifted into describing factory wiring rather than the actual design

## Decision

NeuroX now uses:

- config trees that mirror ownership trees
- owner-constructs-child semantics
- family bases with concrete config subclasses
- `RegistryDispatchMixin` keyed on the concrete config class for impl registry / lookup
- family-specific `from_config(...)` methods with explicit parameters

The mixin does **not** provide a universal `from_config(...)`. Families keep control of their own runtime parameters.

## Consequences

Positive:

- ownership is visible from config structure
- concrete implementation choice is driven by config type
- static typing is tighter
- constructors no longer depend on ad-hoc factory closures

Tradeoff:

- family bases must write their own `from_config(...)`
- parent configs must explicitly carry child config fields
