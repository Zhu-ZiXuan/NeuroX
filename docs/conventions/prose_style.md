# Prose style

Economy, terminology, and punctuation for explanatory text in every medium: markdown prose, docstrings, and inline comments. A docstring or inline comment follows them exactly as markdown prose does. The permitted character set, symbol meanings, the unit set, where a formula lives, and the one-quantity-one-symbol rule all live in [notation_conventions](notation_conventions.md).

## Baseline references

- [Google developer documentation style guide](https://developers.google.com/style/)
- [MathJax documentation](https://docs.mathjax.org/)

Follow these public conventions unless a rule here is stricter or a deviation is noted below.

## Economy

Assume the reader is fluent in the relevant math, physics, circuits, and programming fundamentals; do not explain basic concepts.

- State the conclusion; do not over-justify. Omit the design rationale, the rejected alternative, and the undisputed trade-off.
- Match length to content; do not pad a thin point or inflate text to fill a template.
- State each point once; do not restate it across sections.
- When a name or signature already states a fact — a variable, method, class, or type — do not repeat it in prose.

## Terms

- Reuse the established term for a concept; do not coin a synonym. One concept, one term, with a single home in the [glossary](glossary.md), which also settles contested names such as `snap` over "transient".

## Deviations from the baselines

These differ from the baselines at the character and unit layer:

- **Units** — write a physical unit in prose in ASCII (`uA`, `V`, `MOhm`), matching the code `__unit` suffix; see [notation_conventions §Units](notation_conventions.md#units-and-naming) for the set and the glyph-vs-ASCII rule.
- **Ranges** — join a numeric range with "to", as in `1 uA to 5 uA`, not an en dash.
- **Signs** — use `-` in prose; write a minus sign only inside LaTeX, as `$-x$`.
- **Acronyms** — skip the first-use expansion for a form the audience already knows (ADC, DAC, RAM, SRAM); expand only a genuinely unfamiliar acronym.
