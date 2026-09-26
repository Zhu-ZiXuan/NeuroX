# Writing tests

## Select meaningful coverage

A test should detect a defect owned by the implementation under test and add coverage beyond existing tests of that responsibility. Integration tests qualify when they detect errors in how components are combined.

Use independent oracles: finite differences for derivatives, conservation residuals for solvers, and exact results for sign, phase, and positional-weight assembly. Repeating a formula or the implementation's steps is not an independent check. Thin delegation, generated field assignment, and simple defensive predicates do not need mirror tests unless they participate in meaningful ordering or state-preservation behavior.

Before removing overlapping tests, retain their unique assertions. Similar input values alone do not establish redundancy.

## Inputs and assertions

- Construct test configurations explicitly, with the values relevant to the assertions visible locally. Use presets in campaigns that validate those configured designs.
- Assert observable calling and extension contracts. Do not freeze incidental storage reuse, helper call sequences, or random-number consumption.
- Prefer invariants and independent numerical relationships. Literal expected values are appropriate for exact integer results and specified reference cases.
- Test private helpers directly when they own nontrivial algorithms; privacy alone does not determine test value.
- Inspect compiler graph structure and performance as development diagnostics rather than asserting optimization-dependent details.
- Add runtime validation only where the owning interface requires it; do not replace removed tests with redundant guards.

## Organization

Place behavior tests with the implementation that owns the behavior, following its package path. Cross-component tests belong to the composing owner. Split a large suite by behavior, and keep helpers local unless they are genuinely shared.

Repository-specific static rules live in `tests/rules/`. Use the configured formatter, linter, type checker, and documentation builder for checks those tools already provide.
