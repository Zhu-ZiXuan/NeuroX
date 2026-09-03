# Contributing

Entry point for the NeuroX public development rules. These guides apply equally to human contributors and coding agents.

Prepare and verify a source checkout through [workflow](workflow.md). For a change, start from [recipes](recipes.md), which routes the affected code, tests, docstrings, and documents to their authoritative carriers. [Organizing principles](../conventions/organizing_principles.md) defines those carriers and resolves disagreements between them.

## Documents

- [writing_family_docs](writing_family_docs.md) — the shared Reference science a component family obeys
- [writing_module_docs](writing_module_docs.md) — a single module's Reference document
- [writing_system_design](writing_system_design.md) — a software contract spanning components
- [workflow](workflow.md) — branch / PR flow and quality gates
- [recipes](recipes.md) — task-shape checklists and routing

For writing and coding conventions, start from [conventions](../conventions/README.md). For the scientific specification, start from [reference](../reference/README.md). Mechanical conventions are machine-enforced by `tests/rules/`, whose failure messages cite the governing convention.
