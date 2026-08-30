# Writing module documents

## Scope

A module document states one scheme's own science — the physics and mathematics the scheme obeys — independent of implementation.

A module document holds:

- the physical or circuit model the scheme realizes
- its governing equations
- its numerical method as mathematics
- its own noise and non-ideality sources
- its parameters and their provenance
- its assumptions, scope, validity, validation, and literature

A module document owns science only — not the software that realizes it, its decisions, contracts, or performance.

## Document template

A module document's title is the science name of the scheme it specifies; see [naming_conventions](../conventions/naming_conventions.md). Length follows model content: a section appears only when it carries model content, never padded and never given an invented rationale. A section that is conventionally expected but genuinely empty may be left as a bare `N/A`, or `N/A — <note>` when the note states a real fact about the model (not a description of a consumer), or `TODO — <missing item>` when applicable but unwritten. Sections keep the order below when present.

```markdown
# <Model name>

## Physical model

## Governing equations

## Numerical method

## Noise & non-idealities

## Parameters

## Symbols

## Assumptions, scope & validity

## Validation

## References
```

## Filling each section

An optional lead paragraph directly under the H1 stands in for a Summary section. Write it only when it synthesizes something the body does not — the model's essence and its key runtime inputs; omit it when it would only restate the title or duplicate the body. Do not add a `## Summary` heading.

- `Physical model`: the device, circuit, or architecture physics the model realizes, and the idealizations it deliberately makes.
- `Governing equations`: the equations the model obeys.
- `Numerical method`: the mathematical formulation, well-posedness, and convergence when relevant.
- `Noise & non-idealities`: the non-ideal sources the model implements.
- `Parameters`: the model parameters, their physical constraints, and their provenance.
- `Symbols`: every symbol the document uses.
- `Assumptions, scope & validity`: the modeling assumptions and the range over which they hold.
- `Validation`: how the model is checked against physical data or analytic results.
- `References`: the literature backing the model.

Do not invent physical claims, numbers, equations, validation results, or citations — leave `TODO`.

## Content rules

### Spec the model, not the physical realization

Document what the model computes — its equations, parameters, and modeled non-idealities — and the idealizations it deliberately makes. Do not narrate the physical mechanism the model does not implement, at any layer — device, circuit, or architecture; when the model reduces its subject to a single relation, that relation is the model content and the mechanism narration is not. Keep every idealization and validity statement: what the model deliberately does not model is itself model content.

### Stay in the science layer

A module document states electrical and physical fact only. It carries no software or object-oriented structure, no class or type names in prose, and no architecture or consumer language — no naming of a consuming block, a consuming solve, or an inheritance relation. A plain electrical term, such as a channel-polarity name, is a physical fact and stays as prose without code styling. Code identifiers appear only in the symbol-table Code-field column, nowhere else in the document.

### Equations

- Use standard physics and EE symbols; do not put code identifiers inside equations.
- Coarse-grained equations are allowed when they express the spec better than implementation detail.

### Symbols

Every module document includes a Symbols table listing every symbol it uses, including common ones. Take common symbols from [notation_conventions](../conventions/notation_conventions.md). When a family document provides a shared symbol table, cite it and list only the symbols this scheme adds beyond it.

```markdown
| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
```

### Parameters

Use the five-column table below. Give every parameter a Constraint and a Source; [module_parameter](../conventions/module_parameter.md) defines both columns and what each admits. Runtime inputs such as activations, weights, temperature, and operating points earn no row — define the important ones in the lead paragraph, Governing equations, or Symbols instead.

```markdown
| Parameter | Meaning | Unit | Constraint | Source |
| --- | --- | --- | --- | --- |
```

### Noise

The Noise & non-idealities section states each source's physical or statistical model and its distribution parameters.
