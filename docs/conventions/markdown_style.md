# Markdown style

Structure and format rules for markdown files under `docs/`.

## Baseline references

- [GitHub Flavored Markdown](https://github.github.com/gfm/)
- [Google developer documentation style guide](https://developers.google.com/style/)

Follow these public conventions unless a rule here is stricter or a deviation is noted below.

## Format rules

- File and directory names follow [naming_conventions](naming_conventions.md).
- H1 and every nav label use sentence case (proper nouns, acronyms, and code identifiers stay capitalized).
- Every fenced code block declares a language (stricter than GFM/CommonMark, which make the info string optional).
- Link documents with relative `.md` paths; link sections with `#slug`.
- Never hand-author an HTML anchor (`<a id="..."></a>`); heading anchors are auto-generated, so a `#slug` link needs no manual target, and a link resolves to a whole document.
- README files are navigation only: short orientation plus links to children. Do not put templates, rules, philosophy, or detailed model prose in a README.
- Under `reference/` and `internals/`, only the pillar root has a README. Nested groups exist only in `mkdocs.yml`; do not create a README merely because a directory exists.
- Do not write easy-to-stale counts such as "this family has N members".

## Symbols and math

A formula's symbols go through LaTeX — `$...$` inline, `$$...$$` block; a lone symbol in prose may be a raw whitelisted character (see [notation_conventions](notation_conventions.md)), and everything else is ASCII or LaTeX.

- **Structural math** — roots, powers, fractions, and big operators are LaTeX, where braces delimit scope: `\sqrt{...}`, `e^{x}`, `\frac{a}{b}`, `\sum_{k}`. Never render a 2-D or scoped construct as a raw Unicode glyph.
- **Units** — a physical unit is its ASCII name; see [notation_conventions §Units](notation_conventions.md#units-and-naming).
- **Symbol names** — a variable's source name and its rendered symbol are paired in [notation_conventions](notation_conventions.md); this file does not repeat that correspondence.

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
