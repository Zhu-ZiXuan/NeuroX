# Writing family documents

## Scope

A family document states once the physics and mathematics that every member of a component family obeys, so its members cite it instead of repeating it. It is optional.

A family document holds:

- the family-wide conventions every member obeys — polarity, signed-code convention, floor and quantization semantics
- the governing laws shared across the family, such as a common rescale relation
- the shared symbol table members reference in place of their own
- family-level assumptions and the range over which the shared conventions hold
- family-level literature

A family document owns science only.

## Document template

A family document uses the sections below in order. The document file is `family.md`, titled `<Family> family`; see [naming_conventions](../conventions/naming_conventions.md). The template omits Physical model, Numerical method, Parameters, and Validation, because a family fixes no concrete model, method, or parameter and runs no validation of its own. Only Summary / role and Shared conventions are required; an on-demand section with nothing to say is omitted outright rather than kept as an N/A heading. The footer likewise carries no Configuration or Validation line.

```markdown
# <Family> family

## Summary / role

## Shared conventions

## Governing laws

## Symbols

## Assumptions, scope & validity

## References

---

- **Internals**: [<base doc>](<relative .md path>)
- **Modules**: [README](README.md)
```

## Filling each section

- `Summary / role` (required): what the family is and the shared science it factors out, in role language; describe the shared runtime interface abstractly and never hard-link a concrete member, because the README owns which concretes exist.
- `Shared conventions` (required): the family-wide science every member obeys — polarity, signed-code convention, floor and quantization semantics — stated as science, not as a software contract or a does-not-own boundary.
- `Governing laws` (on-demand): the laws shared across the family, such as a common rescale relation, at coarse math; omit the section when there are none.
- `Symbols` (on-demand): the shared four-column table (Symbol, Meaning, Unit, Code field) members reference instead of repeating, with meanings from [notation_conventions](../conventions/notation_conventions.md); omit when the family shares no symbols.
- `Assumptions, scope & validity` (on-demand): the family-level assumptions and the range over which the shared conventions hold; omit when none are family-wide.
- `References` (on-demand): the family-level literature; leave a `TODO` when it is applicable but unwritten, and omit the section when there is none.

Insert an optional `Noise & non-idealities` section, after Governing laws, only for a statement genuinely shared by every member — for example, that quantization is intrinsic to the whole family. Shared cross-device sources belong in [nonideality](../reference/nonideality.md).

The footer is traceability: Internals points to the base document that owns the family's software contract, and Modules points to the README that indexes the concrete members.

## Content rules

### Stay at the family layer

A family document states shared science, never a software contract and never one member's specifics. An interface obligation, an ownership boundary, or a single topology's transfer characteristic belongs in the base document or that member's document.

**Bad:** "The member does not source or store its reference taps; the owning module injects them each call."

**Good:** "Every member digitizes against an externally supplied reference, so the code edges follow that reference rather than the converter."

The bad version states an ownership contract, which is base-document material; the good version states the shared physical convention.
