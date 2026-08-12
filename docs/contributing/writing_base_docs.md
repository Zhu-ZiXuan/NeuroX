# Writing base documents

## Scope

A base document describes the abstract root a polymorphic module family inherits — the shared contract, never any one implementation. It speaks in family-level role language and names no concrete subclass.

Cross-cutting common infrastructure is documented by its relevant topic page rather than by a file-by-file Internals mirror. Its class docstrings remain limited to direct subclass or host requirements. Which concrete members a module family has belongs in the site navigation, not in its base document.

## Document template

The H1 is `# <Family> base` — the family name followed by `base`; the root base follows the same pattern. The document file is `base.md`, mirroring the package root base `base.py`; see [naming_conventions](../conventions/naming_conventions.md). A base document carries the prescribed sections in the order below and closes with the traceability footer.

Only `Contracts & invariants` is required; an optional lead paragraph under the H1 stands in for a Summary. Every other section is on demand — write it when it carries content and omit it entirely when it does not; never keep it as an `N/A` placeholder. A base therefore has no standing `Performance & resources` / `Gotchas` / `Known limitations` trio: those sections appear only when a genuine family-level note exists.

```markdown
# <Family> base

## Design decisions

## Contracts & invariants

---

- **Reference**: [<family doc>](<relative .md path>), N/A — <reason>, or [<README>](<relative .md path>)
- **Implementation**: `neurox/<...>.py`
- **Tests**: `tests/test_<...>.py` or TODO — <what is missing>
```

## Filling each section

- Lead paragraph [optional]: the family's role and the boundary of what the base does not own, in role language; keep it only when it synthesizes more than the H1, and never hard-link a concrete subclass. Do not add a `## Summary` heading.
- `Design decisions` [on demand, usually present]: the family-level software rationale, told as *why it is so* rather than restated from the class statement — config-type dispatch, an empty-marker `Config` / `Policy`, the membership calls (whether the base itself inherits the root base, and which responsibilities are deliberately left outside it), and the generic or type-parameterized surface.
- `Contracts & invariants` [required]: the substance of the page. State the family-wide guarantee no single symbol carries — what every subclass's implementation must satisfy beyond its signature, and the output, range, and ownership invariants the base holds across the family. Cite a per-symbol contract at its docstring instead of re-listing it. Role language only; no concrete-subclass link.

Add a family-level performance or compile-boundary note only when one genuinely exists at the shared layer; otherwise omit it — never write `N/A at this level`. A base owns no parameters, so it has no `Parameters` section; parameters belong to the concrete scheme and roll up into PPA. If a shared parameter is truly unavoidable, state it in one sentence rather than adding the section.

The footer is traceability. `Reference` links the family's Reference document, or is `N/A — <reason>` for a base with no physics twin, or links the README; `Implementation` lists the base module file(s) as inline code at file level; `Tests` names the guarding test file or a `TODO`.

## Content rules

### Software is the subject

A base document specifies software structure, and that structure is its proper subject; class and Protocol names belong here. It does not re-narrate or restate the family's Reference spec: where a contract rests on a physical fact, state the fact once in Reference and cite it.

### No downward links

A concrete subclass depends on its base, so the base is the lower module and its subclasses are the upper consumers. It must not name or link a concrete subclass in `Summary` or `Contracts & invariants` — pointing a base at its subclass reverses the dependency direction. The site navigation owns the member list; phrase every obligation as a role the subclass fills, not as a named implementation.

This prohibition covers the whole document, including examples, gotchas, and
performance notes. "The current implementation", "for example", and a
hypothetical future member are not exceptions. Put member-specific constructor
arguments, shapes, algorithms, and failure modes in that member's leaf document.

### Contracts & invariants

The base is the single home of the family-wide guarantee: a leaf document states only its own deltas, so every member can rely on one authoritative statement.
