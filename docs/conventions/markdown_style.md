# Markdown style

Structure and format rules for markdown files under `docs/`.

## Baseline references

- [GitHub Flavored Markdown](https://github.github.com/gfm/)
- [Google developer documentation style guide](https://developers.google.com/style/)

Follow these public conventions unless a rule here is stricter or a deviation is noted below.

## Format rules

- File and directory names use `snake_case`.
- H1 and every nav label use sentence case (proper nouns, acronyms, and code identifiers stay capitalized).
- Every fenced code block declares a language (stricter than GFM/CommonMark, which make the info string optional).
- Link documents with relative `.md` paths; link sections with `#slug`.
- README files are navigation only: short orientation plus links to children. Do not put templates, rules, philosophy, or detailed model prose in a README.
- Do not write easy-to-stale counts such as "this family has N members".

## Choosing a format

Match the format to the content, not to habit:

- **Table** — multi-attribute or columnar data: parameters, symbols, constants, comparison or scheme matrices.
- **Numbered list** — ordered procedures, steps, or checklists.
- **Bulleted list** — a scannable set of parallel, independent items: rule sets, contracts and invariants, named-member enumerations (tiers, classes, layers, schemes), definition lists (term — def), risks and trade-offs, nav indexes. Give each item a bold lead-in label when it names a concept; multi-sentence items are fine if they are genuinely parallel members of one enumeration.
- **Prose** — interdependent, connected reasoning: rationale, narrative, a flowing trade-off argument, or a math derivation. Also correct when a section is genuinely one idea.

Test: if the reader benefits from scanning or parallelism → use a list; if the value is in the connective logic → use prose. Never chop one continuous explanation into bullets (prose in disguise), and never bury an enumerable set in a wall of prose.

## Examples

Examples are minimal and illustrative: one short typical snippet per rule. Never paste a full real docstring, parameter table, or method body into a convention file; canonical full text lives at its single source.

## Diagrams

- Structure, flow, and hierarchy diagrams use Mermaid.
- Precise circuit schematics or device cross-sections use vector images under `assets/`. Until an asset exists, leave a `TODO`; never approximate a circuit in ASCII or Mermaid.
