# Prose style

Economy, terminology, punctuation, math, and units for explanatory text in every medium: markdown prose, docstrings, and inline comments. A docstring or inline comment follows them exactly as markdown prose does. The permitted non-ASCII character set and the placement of complex notation live in [notation_conventions](notation_conventions.md).

## Baseline references

- [Google developer documentation style guide](https://developers.google.com/style/)
- [MathJax documentation](https://docs.mathjax.org/)

Follow these public conventions unless a rule here is stricter or a deviation is noted below.

## Economy

Assume the reader is fluent in the relevant math, physics, circuits, and programming fundamentals; do not explain basic concepts.

- State the conclusion; do not over-justify. Omit the design rationale, the rejected alternative, and the undisputed trade-off.
- State each point once; do not restate it across sections.
- When a name or signature already states a fact — a variable, method, class, or type — do not repeat it in prose.

## Terms

- Reuse the established term for a concept; do not coin a synonym. One concept, one term, with a single home in the [glossary](glossary.md), which also settles contested names such as `snap` over "transient".

## Symbols in prose

- `—` sets off a parenthetical or appositive aside.
- `§` prefixes a section reference, e.g. `§Parameters`.
- `×` is multiplication in running text: a multiplier, dimension, or Cartesian product, e.g. `5×`, `64×64`, `batch×inst`.
- `→` marks flow, mapping, or correspondence.
- Write a lone symbol directly in prose; open math only for a real expression, not a single glyph.

## Math

- Use LaTeX: `$...$` inline, `$$...$$` block.
- Named operators use `\operatorname{}`; reserve `\mathrm{}` for upright labels and subscripts.
- Inside math, spell operators as LaTeX commands such as `\to`, `\leftrightarrow`, and `\times`.
- Symbol meanings, the unit set, and the one-quantity-one-symbol rule come from [notation_conventions](notation_conventions.md).
- In comments, do not develop a mathematical or physical derivation; a minimal intent note is fine. Derivations belong in Reference or Internals.

## Deviations from the baselines

These differ from the baselines at the character and unit layer:

- **Units** — write a physical unit in prose in ASCII (`uA`, `V`, `MOhm`; never the micro, ohm, or degree sign), matching the code `__unit` suffix; the ASCII name holds inside math too, never `$\mu\mathrm{A}$` or `$\Omega$`.
- **Ranges** — join a numeric range with "to", as in `1 uA to 5 uA`, not an en dash.
- **Signs** — use `-` in prose; write a minus sign only inside LaTeX, as `$-x$`.
- **Acronyms** — skip the first-use expansion for a form the audience already knows (ADC, DAC, RAM, SRAM); expand only a genuinely unfamiliar acronym.
