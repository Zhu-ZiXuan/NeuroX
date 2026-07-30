# Writing leaf documents

## Scope

A leaf document captures one concrete scheme's own software — one member of a family: the design reasoning, contracts, and engineering trade-offs the code itself cannot show.

A leaf document holds:

- design decisions
- its differences from the base contract, and its own cross-file contracts, ownership, and lifecycle rules
- shape / dtype / state and compile invariants
- complexity, memory model, and performance trade-offs
- gotchas and known limitations

## Document template

A leaf document mirrors a single code module. A section appears only when it carries content and is omitted when it does not — never padded and never kept as an empty heading; use `TODO — <missing item>` only when a section is applicable but unwritten. Sections keep the order below when present.

A leaf document's H1 is verbatim identical to its module document's title; when no module document exists, the H1 names the subject. Its file mirrors the code module basename; see [naming_conventions](../conventions/naming_conventions.md).

```markdown
# <Module name>

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

An optional lead paragraph directly under the H1 stands in for a Summary: one-line orientation. Keep it only when it synthesizes more than the H1; omit it otherwise, and do not add a `## Summary` heading. The matching Reference spec is linked from the footer.

- `Design decisions`: the core section — non-obvious choices, their rationale, and verification-strategy rationale, not ordinary implementation steps.
- `Contracts & invariants`: the cross-file rules and invariants.
- `Performance & resources`: complexity, memory model, chunking, dtype trade-offs, compile boundaries, and benchmark implications.
- `Gotchas`: error-prone behavior and anti-patterns.
- `Known limitations`: implementation TODOs, workarounds, and verification coverage gaps.

The footer is traceability: write Implementation and Tests as inline code at file level only, never a class, function, or line. The Implementation entry is the document's code map, so the body adds no per-file code listing.

## Content rules

### Software is the subject

A leaf document specifies software — decisions, contracts, and trade-offs — and that is its proper subject; class and type names belong here. It does not re-narrate or restate the module's Reference spec; where a decision rests on a physical fact, cite Reference rather than repeating it.

### Contracts & invariants

State call conventions, ownership, shape / dtype / state contracts, lifecycle rules, and compile constraints that hold across files.

### Public contracts

A shared public-interface contract lives at the base class, mixin, or shared concept that owns it, never copied into each consumer. The cross-cutting contract docs are indexed in [internals/README](../internals/README.md). A leaf Internals document states only its module-specific decisions and its differences from the shared contract; it does not restate the contract or maintain a catalog here.

### Do not narrate code

If a sentence only restates control flow or lists files, delete it.

**Bad:** "Each chunk iteration drops `solver_dcop`; its tensors are freed before the next chunk allocates."

**Good:** "Stream by chunk and release the DCOP because the leading batch is large enough to OOM if all node voltages are materialized. Invariant: peak working set stays proportional to one chunk."

The good version gives reason and invariant; the bad version repeats control flow.
