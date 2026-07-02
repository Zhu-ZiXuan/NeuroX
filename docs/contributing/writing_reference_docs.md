# Writing Reference documents

## Scope

Reference is the scientific and engineering golden truth of NeuroX, written in a domain researcher's voice: the mathematical and physical principles, the device and circuit characteristics, and the architecture, dataflow, and algorithm design — all independent of implementation. Core code is a translation of Reference into Python.

Reference holds:

- physical models and device or circuit characteristics
- governing equations
- numerical methods as mathematics
- architecture, dataflow, and algorithm design
- noise and non-ideality models
- parameters and provenance
- assumptions, scope, validity, validation, and literature

Reference mirrors the physics-bearing core library structure to directory granularity; directories are concept groups and extension points.

## Document template

A module document mirrors a single code module and uses every template section below, in order; README files are navigation only. Keep an empty heading as `N/A — <reason>` when genuinely inapplicable or `TODO — <missing item>` when applicable but unwritten; the footer Decisions line may be a bare `N/A` or `None`, as most modules have no ADR.

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
- **Decisions**: [<ADR>](<relative .md path>), N/A, or None
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

The footer is traceability only: Reference points to Internals, Validation, Configuration, and Decisions — it lists no source files or tests, because Internals points to code and tests.

## Content rules

### Equations

- Write equations in LaTeX: `$...$` inline, `$$...$$` block.
- Use standard physics and EE symbols; do not put code identifiers inside equations.
- Coarse-grained equations are allowed when they express the spec better than implementation detail.

### Symbols

Every module document includes a Symbols table listing every symbol it uses, including common ones. Take common symbols from [notation_conventions](../conventions/notation_conventions.md).

```markdown
| Symbol | Meaning | Unit | Code field |
|---|---|---|---|
```

### Parameters

Use a table and give every parameter a Source from [module_parameter](../conventions/module_parameter.md). For calibrated parameters, state whether calibration targets physical data or numerical convergence. Runtime inputs such as activations, weights, temperature, and operating points are inputs, not parameters; define important inputs in Summary, Governing equations, or Symbols.

Cross-cutting physical conventions, constants, units, symbols, and noise notation belong in [notation_conventions](../conventions/notation_conventions.md); cross-cutting terms and concept definitions belong in [glossary](../conventions/glossary.md). A subsystem may explain how a term is used in its own contract but never redefines it.

### Noise

The Noise & non-idealities section describes only each source's physical or statistical model and its distribution parameters — the invariant golden truth. Whether a source is enabled is a runtime policy orthogonal to the model, and how samples are drawn is a program detail; both belong in Internals and configuration, not Reference. Toggles and config parameters are not one-to-one: many sources derive their parameters from physics and carry only a policy toggle, with no config parameter.

## Style

Use concise academic prose. Follow [organizing_principles](../conventions/organizing_principles.md) for present-state-only writing.
