# Writing Reference Documents

Reference is the scientific specification of NeuroX: what is physically and mathematically true. Core code is a translation of these documents. Write for an expert reviewer in devices / circuits / architecture — assume domain expertise, not programming ability.

## Specification, not implementation

- Put here: physical models, governing equations, numerical methods *as mathematics*, noise models, parameters, assumptions, validation, references.
- Put implementation (data layout, shapes, complexity, code structure) in [internals](../internals/README.md).
- Test each sentence: if it stays true after a full code rewrite, it is spec; if it describes the code, move it to internals.
- Do not paraphrase code. Document the physics/math; link the code from the footer.

## Organize by mirroring the code directory tree

- reference/ mirrors `neurox/` **to directory granularity**: every code directory (at any depth) has a corresponding documentation directory. Directories are concept groupings and future extension points (topology families, device classes) — keep them, never flatten a subdirectory.
- **Merge only the direct files of one directory** when they describe one coherent topic (e.g. the `solver/` package's `base` + `nested` + `primitives` become one `solver.md`); never merge across directories, never collapse a subdirectory.
- A directory's abstract layer (its direct files) is the directory-level document(s); subdirectories recurse the same way. Example: `xbar/` = `base.md` (abstract `Xbar` + ideal twin) + `cell.md` (abstract cell) + `solver.md` (the shared `solver/` package merged) + `_1t1r/` (the 1T1R array unit: `cell` `XbarCell1T1R` + `core` `Core1T1R`, co-located because cell and core co-vary by array type). Adding a `_2t1r/` array unit only adds a sibling directory and leaves the abstract layer and the shared solver untouched — mirroring "add the `_2t1r` cell and core, leave the abstract layer alone" in code. Concrete schemes (drivers + reference + coding + readout above a core) live outside the `xbar/` tree under their own scheme directory, mirroring the per-scheme code package.
- reference mirrors only the physics-bearing directories; pure-software structure goes to [internals](../internals/README.md), which mirrors the code tree the same way (spec content here, implementation content there).
- Put cross-cutting physical conventions (constants, units, symbols, noise) in [notation_conventions](../reference/notation_conventions.md), not in a subsystem document.

## Equations

- Write every equation in LaTeX (`$...$` inline, `$$...$$` block). Example: `$$F_{X,k}=I_{\mathrm N}(V_{\mathrm{WL},k},V_{\mathrm{X},k},V_{\mathrm{SL},k})-I_{\mathrm R}(V_{\mathrm{BL},k}-V_{\mathrm{X},k})=0$$`
- Use standard physics / EE symbols ($V$, $I$, $G$, $\mathbf{W}$). Never put a code identifier (`v_bl_node`, `I_R`) inside an equation.
- Coarse-grained equations are fine: write a matmul as `$\mathbf{m}=\mathbf{W}^{\!\top}\mathbf{x}$`.

## Symbols table

- Include a Symbols table listing every symbol the document uses — including common ones; do not omit a symbol because it is common. Columns: symbol | meaning | unit | code field.
- Take common symbols from [notation_conventions](../reference/notation_conventions.md) so one physical quantity keeps one symbol everywhere.

## Document template

Use every section, in order. Never drop a section (see Empty sections).

```text
0. Summary / role      — what physical object this models; where in the device→circuit→architecture stack
1. Physical model      — the reality and the abstraction taken; state core assumptions up front
2. Governing equations — the math, every symbol defined; boundary / initial conditions
3. Numerical method    — the solving scheme as mathematics (well-posedness, convergence); performance and iteration code go to internals
4. Noise & non-idealities — each source: physical origin, statistical model, parameters, policy switch, citation
5. Parameters          — table with a Source column (see below)
6. Assumptions, scope & validity — what is NOT modeled, validity ranges, known limitations
7. Validation          — link to validation/ evidence
8. References          — literature
```

## Parameters section

- Use a table; give every parameter a Source from the [parameter provenance](../reference/parameter_provenance.md) taxonomy.
- For a Calibrated parameter, note in the row whether it is calibrated against physical data or numerical convergence.
- Runtime inputs (activations, weights, temperature, ADC operating point) are inputs, not parameters — do not list them.

## Empty sections

Never omit a section. Keep the heading and write one of:

- `N/A — <why it does not apply>` — genuinely inapplicable (e.g. an exact digital block has no noise section).
- `TODO — <what is missing>` — applies but unwritten.

The empty state is information: a missing §Assumptions or §Validation is a risk flag, not untidiness.

## Footer

End with a horizontal rule and a markdown list — **not** a code block, so the links render. Use relative `.md` links for docs (mkdocs validates and rewrites them); when a target does not exist yet, write `TODO — <what is missing>` instead of a link:

```text
---

- **Internals**: [<doc>](<relative .md path>)
- **Validation**: [<doc>](<relative .md path>)   or   TODO — <what is missing>
- **Configuration**: [<doc>](<api page>)
- **Decisions**: [<ADR>](<relative .md path>)   or   TODO — ...
```

Do not list source files or tests in a reference footer. Reference points to internals; internals points to code and tests.

## Authoring boundary

- Sections 1-5: write from engineering knowledge or re-file existing prose.
- Sections 6-8 and any equation or citation that does not yet exist: do not invent. Leave `TODO — <what is needed>`. Never fabricate a physical claim, a number, or a citation.

## Style

Academic prose for an expert reviewer. English. Concise and precise. Current-state only — no history (that belongs in ADRs). No unnecessary blank lines.
