# Organizing principles

## Readers and tasks

The published site serves library users, model researchers, and extension authors. It starts with installation and a complete library operation, then explains interfaces, models, and validation. Project contributors read the root `CONTRIBUTING.md` and its linked development and writing rules in the repository; these files are excluded from the site.

Choose the form for the reader's task: a tutorial teaches through a runnable example; a guide completes a task; API reference defines a software interface; scientific reference defines a model; a design explanation connects responsibilities across components. Source directories organize authorship; site navigation organizes reading.

## Authoritative sources

| Source | Responsibility |
| --- | --- |
| Root `README.md` | Project purpose, scope, a short starting route, and documentation entry points |
| Root `CONTRIBUTING.md` | Preparing a development checkout, checking changes, and submitting contributions |
| `LICENSE` | License text |
| `CITATION.cff` | Machine-readable citation metadata, when populated |
| `AGENTS.md` | Agent entry instructions and agent-specific requirements; shared rules remain in the contribution guide |
| `pyproject.toml` | Supported dependency ranges, package metadata, dependency groups, and tool configuration |
| `uv.lock` | Resolved development dependencies; it does not define the library's compatibility promise |
| `Makefile` | Repository task implementations |
| Docstrings and field annotations | Calling and extension contracts of the owning package, module, class, method, function, or data field |
| Inline comments | Local implementation intent and non-obvious numerical choices |
| `docs/reference/` | Physics, mathematics, numerical methods, model assumptions, and validity limits |
| `docs/validation/` and `validations/` | Validation methods and the runnable campaigns, configuration, and provenance supporting them |
| `docs/guides/` and `docs/get_started/` | Installation, complete tasks, examples, and choices among interfaces or models |
| `docs/system_design/` | Mechanisms whose responsibilities or observable behavior span components |
| `docs/contributing/` and development conventions | Detailed contribution tasks and project-specific authoring rules |
| Scientific vocabulary under `docs/conventions/` | Shared terms, notation, units, and parameter-source definitions |
| Executable checks | Enforcement of contracts and mechanical conventions |

Standard repository files contain enough information to perform their basic task. Split a topic when it needs independent explanation; do not add a chain of index pages just to relocate a short workflow. Keep contributor instructions in the repository. The site generates API reference from source docstrings without maintaining a second handwritten copy.

## Software contracts beside code

Information needed to call or extend a symbol correctly belongs beside that symbol. The class docstring describes purpose, lifecycle, invariants, and subclass obligations. Method and function docstrings describe operation-specific inputs, outputs, shapes, units, state changes, failures, and override requirements. Field descriptions state meanings and constraints next to their declarations.

Public bases and tools need usable contracts even when their audience is extension authors. A documented extension hook can require detailed explanation despite a leading underscore. Concrete implementations inherit common contracts and document their extra restrictions or behavior. State required parent calls and hook ordering where an author implements the hook.

Use `mkdocstrings` to present these contracts on API pages. Check the rendered inherited methods and field descriptions as well as successful generation. A short family overview can compare implementations and route readers to API and scientific references; it does not reproduce every method contract.

Keep examples in docstrings only when they clarify non-obvious use; omit routine construction and inheritance demonstrations. Flow-control tools may use compact equivalent Python pseudocode with tensor notation for structured values. Put complete workflows and multi-object examples in guides. An isolated helper rarely needs its own Markdown page.

## Scientific reference

Reference specifies what the model computes: its physical laws, mathematical formulation, numerical method, parameters, assumptions, and validity. A family page owns shared science; a concrete model page owns its additional science. A module with no independent scientific content needs no model page.

Mathematical tensor meaning belongs in Reference. Python tensor layout, axis order, dtype, encoding, and device requirements belong in the interface docstring. A restriction needed at the call site may be stated briefly there and explained in the model reference. Preserve idealizations and limitations; do not invent mechanisms, evidence, or results.

Reference length follows model content. Omit empty sections instead of adding filler. Document pointers connect code to its scientific source; equations and derivations remain in Markdown.

## Cross-component explanations

A system design page explains a question that requires understanding multiple responsibilities: physical-state lifetimes, sampling correlation, scheduling, or accounting ownership. It serves users interpreting results as well as extension authors and maintainers.

A contract wholly owned by one abstraction belongs in that abstraction's docstring. A guide may demonstrate how several abstractions work together. Explain design reasons and trade-offs when they affect a user's choice or an extension author's implementation. Detailed source walkthroughs belong near the implementation.

## Single source and useful summaries

Each rule or definition has one authoritative source. Update it first, then align dependent code, checks, examples, and summaries. Configuration defines supported settings; prose explains their use. Observed behavior does not silently replace a scientific specification.

Concise summaries, installation commands, examples, and call-site constraints are allowed where readers need them. Link to the full definition and verify examples when their interfaces change. Avoid copying complete contracts, configuration tables, or API catalogs into multiple handwritten pages.

Prefer established public standards for general writing and tooling. Local conventions record NeuroX-specific requirements and deliberate differences. Mechanical formatting belongs in tool configuration where possible.

## Scope and navigation

Describe a reusable component through its own responsibilities. An upper component may explain how it composes lower ones. Examples, guides, and comparisons may name concrete implementations to help readers choose and use them; label example-specific choices as such. Documentation examples do not change code dependency rules.

Use links where they help the current task. Directory indexes and site navigation offer overview routes, while a focused page can link prerequisites and related explanations. Every relative link must resolve in its intended source or rendered context.

Current manuals describe implemented behavior. Release history belongs in release notes. Keep unpublished proposals out of the current API and model reference. Explain current design rationale when it helps readers understand supported behavior.
