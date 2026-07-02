# Writing mixin documents

## Scope

A mixin document mirrors one mixin class — a pure-software cross-cutting mechanism, a reusable capability composed into a host. It lives under `internals/common/mixin/`. A mixin document specifies the contract between the mixin and its hosts — the surface a host must provide, the hook a subclass overrides, and the lifecycle point at which the behavior fires — in role language, and names no concrete host.

## Document template

A mixin document carries the sections below in order and closes with the traceability footer. The H1 is `# <Mixin> mixin` — the mixin's name followed by `mixin` (`# Fabricate mixin`).

Only `Summary` and `Contracts & invariants` are required; every other section is on demand — write it when it carries content and omit it entirely when it does not, never as an `N/A` placeholder. A mixin owns no parameters, so it has no `Parameters` section.

```markdown
# <Mixin> mixin

## Summary

## Design decisions

## Contracts & invariants

### Host requirements

### Override surface

## Composition

---

- **Reference**: N/A — software mechanism
- **Implementation**: `neurox/common/mixin/<...>.py`
- **Tests**: `tests/test_<...>.py` or TODO — <what is missing>
- **Decisions**: [<ADR>](<relative .md path>), N/A, or None
```

## Filling each section

- `Summary` [required]: the behavior the mixin grants a host and the boundary of what it does not do, in role language; name no concrete host. Link the cross-cutting lifecycle document when the granted behavior is one phase of a wider lifecycle.
- `Design decisions` [on demand, usually present]: the software rationale for the mechanism — the cascade or dispatch shape, what the host owns versus the mixin, and any rejected alternative.
- `Contracts & invariants` [required]: the payload — the mixin-to-host contract, in the two subsections below; state the inherited method's semantics and any cross-call invariant, such as re-callability without state accumulation, in the section body.
    - `### Host requirements`: the base the host must also inherit and the attributes it must set for the mixin to function.
    - `### Override surface`: the hook a subclass overrides — its signature and its default-body semantics.
- `Composition` [on demand]: the mixin's position in the host MRO and its trigger timing — the lifecycle point at which its behavior fires.

The footer is traceability. `Reference` is always `N/A — software mechanism`; `Implementation` lists the mixin module file as inline code at file level; `Tests` names the guarding test file or a `TODO`; `Decisions` links the governing ADR, or is `N/A` or `None`.

## Content rules

### No downward links

A host composes a mixin, so the mixin is the lower module and its hosts are the upper consumers. State every requirement as a role the host fills — the base it must inherit, the attribute it must set — never as a named host module. Pointing a mixin at a specific host reverses the dependency direction fixed in [organizing_principles](../conventions/organizing_principles.md).

**Bad:** "`OpAmpTia` sets `_inst_shape` before the cascade runs."

**Good:** "The host sets `_inst_shape` in its `__init__` before the cascade runs."

### Single contract home

The mixin document is the single home of the mixin's contract. A host or subclass document states only how it composes or specializes the mixin and never restates the host requirements or override semantics; it links here instead.
