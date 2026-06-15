# Documentation Style

Format rules for every document under `docs/`. Code style is in [code_style](code_style.md); this file covers prose, markdown, characters, and notation rendering.

## Characters

Write ASCII. Use unicode only from this whitelist, only for the stated purpose:

| Char | Use |
|---|---|
| `—` | em dash: parenthetical / appositive break in prose |
| `§` | section sign: cross-reference to a document section (e.g. `§Parameters`) |
| `×` | multiplication in text: multiplier, dimension, or Cartesian product (`5×`, `64×64`, `batch×inst`) |
| `→` | arrow in text: flow or mapping (`A→B`) |

Everything else is ASCII or LaTeX:

- **Physical units**: ASCII, matching the code suffixes — `uA`, `uS`, `um`, `MOhm`, `fF`, `ns`, `K`, `fJ`, `uW`. Never the micro sign or ohm sign in prose; in an equation use LaTeX (`$\mu\mathrm{A}$`, `$\Omega$`).
- **Math symbols** (operators, Greek letters, sub/superscripts): LaTeX `$...$` — `$\approx$`, `$\pm$`, `$10^5$`.
- **No emoji**. Use `**Bad:**` / `**Good:**` labels for examples.
- No en dash (use `-`), no ellipsis character (use `...`), no middle dot (use a comma or a list).

## Math

- Equations in LaTeX: `$...$` inline, `$$...$$` block (mkdocs renders via MathJax).
- Named operators use `\operatorname{}`, not `\mathrm{}`: `$\operatorname{TIA}(\cdot)$`, `$\operatorname{clamp}(\cdot)$`. Reserve `\mathrm{}` for upright subscripts and labels (`$V_{\mathrm{BL}}$`).
- Inside math use `\to` and `\times`. In text use the whitelisted `→` / `×` directly; do not open a `$...$` for a single symbol.
- Use the symbols pinned in [notation_conventions](../reference/notation_conventions.md): one physical quantity, one symbol.

## Files and structure

- File and directory names use `snake_case` (`writing_reference_docs.md`, `algorithm_engineer/`), matching the Python source convention.
- Every fenced code block declares a language (` ```python `, ` ```toml `, ` ```text `).
- Present structured content (parameters, comparisons, options) as tables.
- Link documents with relative `.md` paths (mkdocs validates and rewrites them); link a section with `#slug`.
- A README is **navigation only**: one sentence of orientation plus links to its children. Rules, philosophy, and templates belong in convention or concept documents, not in a README.

## Cross-references and dependency direction

Cross-references go one direction: a specific document references a more general one, never the reverse.

- **Convention and term documents** (this file, `code_style`, `naming_conventions`, `writing_reference_docs`, `writing_internals_docs`, `notation_conventions`, `parameter_provenance`) are the **sink**: every specific document links to them; they link to nothing more specific — no concept document, no subsystem document. When such a document must mention a mechanism, name it in inline code (e.g. `FabricateMixin`); do not link or path-reference a more detailed document.
- **`recipes`** is the one exception: as a task index it is the most downstream document, so it may reference the conventions and concepts it sequences (but not specific subsystem instances).
- The same rule for code-level docs (a lower module's docs never name an upper module) is in [code_style](code_style.md).
- **Do not duplicate the README's catalog.** Reference a sibling document only where the prose needs it — an in-context link at the relevant sentence, or one sentence stating this document's scope versus a sibling's. Do not list sibling documents in a header or footer; that catalog is the README's job. (The `reference` / `internals` footer is not a catalog — it is the spec / impl / code / test / decisions traceability chain.)

## Examples

An example is illustrative and minimal: it shows the rule, not the whole truth. Do not paste a full real docstring, an entire parameter table, or a complete method block as an "example" — use one short, typical snippet per rule. Canonical full text (such as a repeated docstring) has a single source, the code; do not duplicate it into a convention document.

## Footers

End reference and internals documents with a markdown list (not a code block), per [writing_reference_docs](writing_reference_docs.md) and [writing_internals_docs](writing_internals_docs.md).

## Diagrams

- Structure / flow / hierarchy diagrams: mermaid (` ```mermaid `) — text-based and diffable.
- Precise circuit schematics / device cross-sections: a vector image under `assets/`, referenced from the document. Until one is provided, leave a `TODO` placeholder; do not approximate a circuit in ASCII or mermaid.
