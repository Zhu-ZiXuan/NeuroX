# Organizing principles

Knowledge is carried in three media: markdown documents, in-code text (docstrings, comments, shape and type annotations), and executable checks (types, runtime guards, and tests).

## What each carrier holds

One question routes every statement:

> Can one clear code abstraction wholly own this knowledge? If it can, the knowledge belongs to that abstraction — its structure, types, docstring, comments, validation, and tests. If no abstraction can own it and maintaining it requires understanding several components at once, it belongs to System Design. If it stays true after the whole implementation is rewritten, it belongs to Reference.

| Carrier | Holds |
| --- | --- |
| `docs/reference/**.md` | Researcher-voice scientific, mathematical, circuit, architecture, and algorithm principles and design; the golden truth code translates; independent of the concrete implementation |
| `docs/validation/**.md` | The method by which a model and its implementation are shown faithful, and the route to the evidence the repository's campaign directory holds |
| `docs/system_design/**.md` | Cross-component software design: a system contract no single code abstraction owns, which several components must be read together to maintain |
| `docs/contributing/**.md` | How to write documentation and code |
| `docs/conventions/**.md` | The standards every document and source file follows, plus the symbols, terms, and parameter sources shared across subsystems |
| docstring | The contract of the symbol it sits on: caller-facing semantics of a public API, and the extension contract a base, mixin, or protocol places on a subclass author, including the field shapes of a public tensor container |
| inline comment | Local implementation intent — why a line or block is written the way it is |
| shape annotation | The terminal shape a class-header declaration carries, and semantic tensor-shape transitions at the point of code; its format is in [code_style](code_style.md) |
| banner comment | Declaration and field grouping at class scope, and procedural phase boundaries in a method body; its format is in [code_style](code_style.md) |
| executable check | Any rule a type, a runtime guard, a `tests/rules/` repository scan, or a behavior test can decide; such a rule is enforced there rather than restated as prose, and its normative text stays in Conventions |

The code is itself the carrier of what it already states — a signature, a type, a `raise NotImplementedError`. No document restates such a fact; each holds only the reason the code cannot show.

A symbol-level extension contract belongs to the docstring beside the enforcing code — the base, mixin, or protocol that injects the behavior — and covers injected behavior, lifecycle, host requirements, ownership visibility, and failure conditions. No separate page duplicates it.

A runtime variable is program state, not an invariant characteristic: it belongs to the code that owns it, while Reference states the characteristics themselves, as fact. The same split governs tensor shape — Reference owns scientific and mathematical tensor meaning, while implementation layout, broadcast, batching, and reshape invariants belong to the code that performs them, and reach System Design only when the layout contract spans components with no single owner.

## Specify the model, not the physical realization

Reference specifies the model, not the physical realization it abstracts — a device, a circuit, or a whole architecture. Document what the model computes — its governing equations, parameters, and modeled non-idealities — together with the idealizations it deliberately makes; an idealization or validity statement is itself model content. Do not narrate the underlying mechanism the model does not implement, at any layer.

Length follows model content, not the template. A trivial idealized primitive earns a few lines; a rich model earns its length. A section with no model content is omitted, not padded and not filled with an invented rationale; a bare "N/A" suffices, extended only when the note itself carries real model information.

## Directory layout

Each top-level `docs/` directory owns one kind of content:

- `get_started/` — installation and quickstart.
- `guides/` — task how-to, grouped by engineer audience.
- `reference/` — the scientific spec, by subsystem.
- `validation/` — evidence that the models are faithful and correctly implemented, including the per-paper calibration campaigns kept in the repository's `validations/` directory.
- `api/` — the Python API and the config and policy schema.
- `system_design/` — the cross-component software contracts, by topic.
- `conventions/` — documentation and coding standards, plus the shared vocabulary.
- `contributing/` — how to write and place documentation, plus the contribution workflow.
- `about/` — citation and the simulator's scope and limitations.

Reference is grouped by subsystem at directory granularity, so a subsystem's science sits at the path matching the physics-bearing code it specifies; a module carrying no science of its own gets no page. A directory is a concept grouping and an extension point: a new science-bearing subsystem adds a directory of its own name, carrying a README index beside the subsystem's documents.

## System design

A System Design page protects a contract that spans components. Content is admitted only when most of the following hold: it spans several components, layers, or lifecycle stages; no class, protocol, function, or file can wholly own it; changing one end requires understanding the other; it concerns ownership, state, layout, sampling correlation, compile boundaries, or global registration; types and unit tests cannot express it; it is stable; and losing it would materially raise the rate of cross-module error. Two module names appearing in one sentence is not admission on its own, and an interaction that one abstraction already defines and enforces stays in that abstraction's docstring and tests.

Pages are organized by the question a maintainer arrives with, not by the code tree: no mirroring of the code layout, no page per class, no coverage target, no template, and no fixed footer. A page's length follows the contract it protects, and an absent topic is simply absent, never a placeholder section. The category README is a small map of the topics, not a member catalog.

## De-specific voice

- A concrete implementation name is forbidden unless the document is strictly bound to that implementation.
- Strict binding means one of: the implementation is the document's subject; the documented code imports, constructs, inherits, registers, or type-constrains it; an API index inventories it; the site navigation inventories it; or an executable example deliberately instantiates it.
- A default choice, the only implementation presently available, a convenient example, a comparison, or a possible future implementation is not strict binding.
- A top-level or cross-cutting document speaks in role language. It names no concrete consumer, device, module, class, or function unless that symbol is itself the documented public API.
- A document about a shared layer names its own shared symbols but no concrete member; the concrete detail lives with the document or docstring that owns that member.
- An executable example may name the implementation it constructs, but its prose must present that choice as local to the example rather than as a project-wide dependency.

This rule keeps change propagation aligned with code dependencies. Removing one implementation should normally affect only its own documents, its subsystem index, its API export, and its dedicated tests.

## Dependency direction

Documentation follows code dependency direction:

- An upper module may describe how it owns, constructs, calls, or configures lower modules.
- A lower module describes only its own responsibility, interface, invariants, and the lower modules it owns.
- A lower module must not name, assume, or instruct a specific upper consumer.
- No document names an upper-layer usage scenario or constrains how its subject is used; it states the subject's own behavior in neutral terms, e.g. "reused across runs".

This holds in every carrier alike. One example each way:

- **Allowed**: a composite block's document describes how it constructs and programs the leaf cells it owns.
- **Forbidden**: a leaf cell's document states that the composite owning it calls the cell during a forward pass.

The same direction binds the code itself: a package imports its own layer or a lower one, never an upper one (machine-checked in `tests/rules/`).

## Single source

- Each contract, rule, or definition has exactly one authoritative home; everywhere else links to that home instead of copying it.
- Knowledge stated at a broader scope is cited from the narrower one, never restated there: a shared law belongs to the document that owns the shared layer, and a member cites it.
- Project-level rule documents must not maintain subsystem catalogs.
- Name an area briefly; link a specific shared contract only when a sentence depends on it.
- Meta-knowledge about the documentation system itself — the carrier split, the placement criterion, the document categories, and their relationships — lives only in this document; every other document keeps to its own subject.
- A detail document links to a topical rule's single home, but complies with this document without linking to it: it is reached top-down through README alone.

## Links and navigation

- Navigation and index linking belong to README and recipes alone — the sole index and navigation exception to single source and dependency direction. No other document carries a topic-link index or navigation menu in its header; each opens directly on its own subject.
- A reader follows every in-prose link, and each one recursively. Place an in-prose guiding link only when its target genuinely holds information both relevant to the sentence and correct there; otherwise omit it.
- A traceability link is ordinary content: write it where the relation is real and the reader needs it, in the prose that depends on it. No document carries a fixed related-document block, and no relation is filled with a placeholder when the document has none.
- Every relative link resolves to a file that exists (machine-checked in `tests/rules/`).

## Present state only

- Every carrier describes the current state only — no history and no future.
- Remove the entire out-of-state statement, not just its trigger word: delete what the subject once was, what it is not, and what it will become, and state the positive current fact. Never soften "X is no longer Y" into "X is not Y".
- A historical lesson, a rejected or unimplemented alternative, and a one-off implementation decision enter no carrier at all; the code and its tests carry the decision that was taken.
