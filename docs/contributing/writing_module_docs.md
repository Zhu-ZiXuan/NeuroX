# Writing model documents

A model Reference page describes what is computed: physics, mathematics, numerical method, parameters, assumptions, and evidence. Python signatures, tensor layouts, lifecycle requirements, and failure behavior belong in source docstrings.

## Content

Use sections that serve the model; there is no minimum length or required empty template.

- **Physical model:** the modeled mechanism and its idealizations.
- **Governing equations:** the relations that define its outputs.
- **Numerical method:** the mathematical solution procedure, where nontrivial.
- **Noise and non-idealities:** source distributions, dependencies, and lifetimes.
- **Parameters and symbols:** definitions needed to interpret the equations.
- **Validity:** supported assumptions and known limitations.
- **Validation and references:** available checks and the sources supporting the model.

Omit empty `N/A` sections and generic citation placeholders. Preserve specific gaps in evidence or characterized validity; do not replace them with invented claims. Link existing checks when they cover the model, distinguishing numerical consistency from independent physical validation.

## Equations and terminology

Use mathematical symbols in equations and define them under the shared [notation](../conventions/notation_conventions.md). Code identifiers may connect parameter and symbol tables to their implementation; keep configuration syntax and API instructions in their respective guides.

Use the shared family definitions where applicable. A model-specific symbol table lists additions rather than repeating an entire family table:

```markdown
| Symbol | Meaning | Unit | Code field |
| --- | --- | --- | --- |
```

## Parameters

State physical constraints and parameter provenance using the [source taxonomy](../conventions/module_parameter.md):

```markdown
| Parameter | Meaning | Unit | Constraint | Source |
| --- | --- | --- | --- | --- |
```

Runtime operands, temperature, and operating points are inputs to the equations rather than design parameters. Describe their meaning where they enter the model. For a derived parameter, identify the inputs and derivation instead of assigning it an independent source.
