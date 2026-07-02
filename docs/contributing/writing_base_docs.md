# Writing base documents

## Scope

A base document describes the abstract root a polymorphic family inherits. It specifies the surface every subclass inherits and the obligations every subclass must meet: the shared contract, never any one implementation. It speaks in family-level role language and names no concrete subclass.

The same template covers a family base (`# ADC base`) and the root `# Circuit base`, the fixed base every electrical-circuit module inherits. Which concrete members a family has belongs in the family README, not here.

## Document template

The H1 is `# <Family> base` — the family name followed by `base` (`# ADC base`, `# TIA base`); the root base follows the same pattern as `# Circuit base`. A base document carries the prescribed sections in the order below and closes with the traceability footer.

Only `Summary` and `Contracts & invariants` are required. Every other section is on demand — write it when it carries content and omit it entirely when it does not; never keep it as an `N/A` placeholder. A base therefore has no standing `Performance & resources` / `Gotchas` / `Known limitations` trio: those sections appear only when a genuine family-level note exists.

```markdown
# <Family> base

## Summary

## Design decisions

## Contracts & invariants

## Composition

---

- **Reference**: [<family doc>](<relative .md path>), N/A — <reason>, or [<README>](<relative .md path>)
- **Implementation**: `neurox/<...>.py`
- **Tests**: `tests/test_<...>.py` or TODO — <what is missing>
- **Decisions**: [<ADR>](<relative .md path>), N/A, or None
```

## Filling each section

- `Summary` [required]: the family's role and the boundary of what the base does not own, in role language. Link the family's Reference document when one exists and route the reader to the README for the concrete members; never hard-link a concrete subclass.
- `Design decisions` [on demand, usually present]: the family-level software rationale — config-type dispatch, an empty-marker `Config` / `Policy`, the membership call (whether the base is itself a `CircuitBase`), the generic or type-parameterized surface, and any rejected alternative.
- `Contracts & invariants` [required]: the payload. State the shared surface the base provides, the abstract obligations each subclass must implement (signature and semantics), and the invariants the base guarantees. Role language only; no concrete-subclass link.
- `Composition` [on demand]: the mixin stack and MRO order, and the membership boundary — what is inside the base and what is deliberately out.

Add a family-level performance or compile-boundary note only when one genuinely exists at the shared layer; otherwise omit it — never write `N/A at this level`. A base owns no parameters, so it has no `Parameters` section; parameters belong to the concrete scheme and roll up into PPA. If a shared parameter is truly unavoidable, state it in one sentence rather than adding the section.

The footer is traceability. `Reference` links the family's Reference document, or is `N/A — <reason>` for a base with no physics twin, or links the README; `Implementation` lists the base module file(s) as inline code at file level; `Tests` names the guarding test file or a `TODO`; `Decisions` links the governing ADR, or is `N/A` or `None`.

## Content rules

### No downward links

A concrete subclass depends on its base, so the base is the lower module and its subclasses are the upper consumers. It must not name or link a concrete subclass in `Summary` or `Contracts & invariants` — pointing a base at its subclass reverses the dependency direction fixed in [organizing_principles](../conventions/organizing_principles.md). Route the reader to the family README for the member list, and phrase every obligation as a role the subclass fills, not as a named implementation.

### Contracts & invariants

The base is the single home of the family's shared contract; a leaf document states only its own deltas and never restates the contract. Put here every obligation and invariant common to the family — the construction signature, the abstract method surface with its semantics, and the guaranteed output, range, and ownership invariants — so each leaf can rely on this one authoritative statement.
