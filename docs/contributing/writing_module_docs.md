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

A module document mirrors a single code module and uses every template section below, in order. Its file mirrors the code module basename and its title is the science name; see [naming_conventions](../conventions/naming_conventions.md). Keep an empty heading as `N/A — <reason>` when genuinely inapplicable or `TODO — <missing item>` when applicable but unwritten.

```markdown
# <Model name>

## Summary / role

## Physical model

## Governing equations

## Numerical method

## Noise & non-idealities

## Parameters

## Symbols

## Assumptions, scope & validity

## Validation

## References

---

- **Internals**: [<doc>](<relative .md path>)
- **Validation**: [<doc>](<relative .md path>) or TODO — <what is missing>
- **Configuration**: [<doc>](<relative .md path>) or TODO — <what is missing>
```

## Filling each section

- `Summary / role`: what the component is, its role in the system, and the key runtime inputs.
- `Physical model`: the device or circuit physics the spec models.
- `Governing equations`: the equations the model obeys.
- `Numerical method`: the mathematical formulation, well-posedness, and convergence when relevant.
- `Noise & non-idealities`: the non-ideal sources present.
- `Parameters`: the model parameters and their provenance.
- `Symbols`: every symbol the document uses.
- `Assumptions, scope & validity`: the modeling assumptions and the range over which they hold.
- `Validation`: how the model is checked against physical data or analytic results.
- `References`: the literature backing the model.

Never omit a required section; the empty state is information. Do not invent physical claims, numbers, equations, validation results, or citations — leave `TODO`.

The footer is traceability only: Reference points to Internals, Validation, and Configuration — it lists no source files or tests, because Internals points to code and tests.

## Content rules

### Equations

- Use standard physics and EE symbols; do not put code identifiers inside equations.
- Coarse-grained equations are allowed when they express the spec better than implementation detail.

### Symbols

Every module document includes a Symbols table listing every symbol it uses, including common ones. Take common symbols from [notation_conventions](../conventions/notation_conventions.md). When a family document provides a shared symbol table, cite it and list only the symbols this scheme adds beyond it.

```markdown
| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
```

### Parameters

Use a table and give every parameter a Source from [module_parameter](../conventions/module_parameter.md). For calibrated parameters, state whether calibration targets physical data or numerical convergence. Runtime inputs such as activations, weights, temperature, and operating points are inputs, not parameters; define important inputs in Summary, Governing equations, or Symbols.

### Noise

The Noise & non-idealities section states each source's physical or statistical model and its distribution parameters.
