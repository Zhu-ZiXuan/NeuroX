# Writing leaf documents

## Scope

A leaf document captures one concrete scheme's own software — one member of a family: the design reasoning, contracts, and engineering trade-offs the code itself cannot show.

A leaf document holds:

- design decisions and rejected alternatives
- its differences from the base contract, and its own cross-file contracts, ownership, and lifecycle rules
- shape / dtype / buffer and compile invariants
- complexity, memory model, and performance trade-offs
- gotchas and known limitations

## Document template

A leaf document mirrors a single code module and uses every template section below, in order. Keep an empty heading as `N/A — <reason>` when genuinely inapplicable or `TODO — <missing item>` when applicable but unwritten.

A leaf document's H1 is verbatim identical to its module document's title; when no module document exists, the H1 names the subject. Its file mirrors the code module basename; see [naming_conventions](../conventions/naming_conventions.md).

```markdown
# <Module name>

## Summary

## Design decisions

## Contracts & invariants

## Performance & resources

## Gotchas

## Known limitations

---

- **Reference**: [<doc>](<relative .md path>) or N/A — <reason>
- **Implementation**: `neurox/<...>.py`
- **Tests**: `tests/test_<...>.py` or TODO — <what is missing>
```

## Filling each section

- `Summary`: one-line orientation, plus the matching Reference spec when one exists.
- `Design decisions`: the core section — non-obvious choices, their rationale, rejected alternatives, and verification-strategy rationale, not ordinary implementation steps.
- `Contracts & invariants`: the cross-file rules and invariants.
- `Performance & resources`: complexity, memory model, chunking, dtype trade-offs, compile boundaries, and benchmark implications.
- `Gotchas`: error-prone behavior and anti-patterns.
- `Known limitations`: implementation TODOs, workarounds, and verification coverage gaps.

Never omit a required section; the empty state is information.

The footer is traceability: write Implementation and Tests as inline code at file level only, never a class, function, or line. The Implementation entry is the document's code map, so the body adds no per-file code listing.

## Content rules

### Contracts & invariants

State call conventions, ownership, shape / dtype / buffer contracts, lifecycle rules, and compile constraints that hold across files.

### Public contracts

A shared public-interface contract lives at the base class, mixin, or shared concept that owns it, never copied into each consumer. The cross-cutting contract docs are indexed in [internals/README](../internals/README.md). A leaf Internals document states only its module-specific decisions and its differences from the shared contract; it does not restate the contract or maintain a catalog here.

### Do not narrate code

If a sentence only restates control flow or lists files, delete it.

**Bad:** "Each chunk iteration drops `solver_dcop`; its tensors are freed before the next chunk allocates."

**Good:** "Stream by chunk and release the DCOP because the leading batch is large enough to OOM if all node voltages are materialized. Invariant: peak working set stays proportional to one chunk."

The good version gives reason and invariant; the bad version repeats control flow.
