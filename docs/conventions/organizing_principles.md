# Organizing principles

Documentation is carried in two media: markdown files and in-code text (docstrings, comments, shape and type annotations).

## What each carrier holds

| Carrier | Holds |
|---|---|
| `docs/reference/**.md` | Researcher-voice scientific, mathematical, circuit, architecture, and algorithm principles and design; the golden truth code translates; independent of the concrete implementation |
| `docs/internals/**.md` | Engineer-voice Python system design; decisions, trade-offs, and cross-file contracts not readable from code |
| `docs/contributing/**.md` | How to write documentation and code |
| `docs/conventions/**.md` | The standards every document and source file follows, plus the symbols, terms, and parameter sources shared across subsystems |
| docstring | Caller-facing API semantics of the symbol |
| inline comment | Local implementation intent |
| shape annotation | Semantic tensor-shape transitions at the point of code; its format is in [code_style](code_style.md) |
| step comment | Procedural phase boundaries aligned to a numbered procedure in Reference or Internals; its format is in [code_style](code_style.md) |

When a statement could fit two carriers: survives a code rewrite → Reference; describes the program → Internals; caller-facing → docstring; local intent → comment. A runtime variable is program state, not an invariant characteristic; it belongs to Internals. Reference states the characteristics themselves, as fact.

The code is itself the carrier of what it already states — a signature, a type, a `raise NotImplementedError`. No document restates such a fact; each holds only the reason the code cannot show.

The same split governs tensor shape: Reference owns scientific or mathematical tensor meaning; Internals owns implementation layout, broadcast, batching, and reshape invariants.

## Directory layout

Each top-level `docs/` directory owns one kind of content:

- `get-started/` — installation and quickstart.
- `guides/` — task how-to, grouped by engineer audience.
- `reference/` — the scientific spec, by subsystem.
- `validation/` — evidence that the models are faithful and correctly implemented.
- `api/` — the Python API and the config and policy schema.
- `internals/` — the software implementation companion to Reference, by subsystem.
- `conventions/` — documentation and coding standards, plus the shared vocabulary.
- `contributing/` — how to write each document class, plus the contribution workflow.
- `about/` — citation and the simulator's scope and limitations.

Reference mirrors the physics-bearing core library and Internals the software core library, each to directory granularity, so a subsystem's science and its implementation sit at matching paths. A directory is a concept grouping and an extension point: a new subsystem adds a directory of its own name under both trees, each carrying a README index beside the subsystem's documents.

## Document classes

Reference and Internals hold three classes of document, told apart by what they mirror and what they carry:

- A **module-mirroring document** covers one science-bearing code module. It carries the prescribed template sections and the traceability footer, and it participates in the Reference-Internals correspondence, so the 2×2 below classifies it.
- A **mixin document** covers one pure-software mechanism class. It likewise carries a prescribed template and footer, but with no physics to specify it has no Reference twin, so it stands alone in Internals, outside the 2×2. Write it per [writing_mixin_docs](../contributing/writing_mixin_docs.md).
- A **cross-cutting document** spans modules and sits at the Reference or Internals root. It carries neither a template nor a footer and organizes freely around its own content, so it too sits outside the 2×2.

Two orthogonal axes classify a module-mirroring document. The tree is the authority boundary: a statement that survives a code rewrite is Reference, the scientific spec that owns physics and math; one that describes the program is Internals, the software implementation that owns contract and design. The kind is the layer: the shared layer is the abstraction every member holds in common, and the concrete layer is one scheme.

| | Shared layer | Concrete |
|---|---|---|
| **Reference** (science) | family document | module document |
| **Internals** (software) | base document | leaf document |

- **Family document** — the science a whole family shares: conventions, shared laws, and shared symbols, so a concrete member cites it instead of repeating it. Write it per [writing_family_docs](../contributing/writing_family_docs.md).
- **Module document** — one scheme's own science: its physical model, transfer characteristic, energy model, and parameters. Write it per [writing_module_docs](../contributing/writing_module_docs.md).
- **Base document** — the shared software contract: the surface the base provides and the obligations a subclass must implement. Write it per [writing_base_docs](../contributing/writing_base_docs.md).
- **Leaf document** — one scheme's own software: its decisions, its differences from the base contract, and its performance and gotchas. Write it per [writing_leaf_docs](../contributing/writing_leaf_docs.md).

Each tree names the layers in its own domain's words — the Internals inheritance tree uses base and leaf, the Reference scientific taxonomy uses family and module — for the same shared and concrete layers.

These rules hold across the templates:

- **Existence is asymmetric** — a Reference document, family or module, exists only for a science-bearing module, so a base can stand in the 2×2 with no family twin. A family document is optional, written only when the shared science is substantial; otherwise each member carries its own.
- **Layering** — a family document holds only the science shared within one family; cross-device shared science, such as Pelgrom mismatch and $kT/C$ noise, stays at [nonideality](../reference/nonideality.md) and is referenced, not restated.

## De-specific voice

- A top-level or cross-cutting document speaks in general role language: it names no specific consumer, device, module, or function and parades no example drawn from one.
- The specifics live in the module document that owns them.

## Dependency direction

Documentation follows code dependency direction:

- An upper module may describe how it owns, constructs, calls, or configures lower modules.
- A lower module describes only its own responsibility, interface, invariants, and the lower modules it owns.
- A lower module must not name, assume, or instruct a specific upper consumer.
- No document names an upper-layer usage scenario or constrains how its subject is used; it states the subject's own behavior in neutral terms, e.g. "reused across runs".

This holds for docstrings, comments, Reference, and Internals alike. One example each way:

- **Allowed**: a composite block's document describes how it constructs and programs the leaf cells it owns.
- **Forbidden**: a leaf cell's document states that the composite owning it calls the cell during a forward pass.

## Single source

- Each contract, rule, or definition has exactly one authoritative home; everywhere else links to that home instead of copying it.
- Project-level rule documents must not maintain subsystem catalogs.
- Name an area briefly; link a specific shared contract only when a sentence depends on it.
- Meta-knowledge about the documentation system itself — its taxonomy, class names, naming choices, and cross-class relationships — lives only in this document; every other document keeps to its own subject.
- A detail document links to a topical rule's single home, but complies with this document without linking to it: it is reached top-down through README alone.

## Links and navigation

- Navigation and index linking belong to README and recipes alone — the sole index and navigation exception to single source and dependency direction. No other document carries a topic-link index or navigation menu in its header; each opens directly on its own subject.
- A reader follows every in-prose link, and each one recursively. Place an in-prose guiding link only when its target genuinely holds information both relevant to the sentence and correct there; otherwise omit it.
- Footer links are exempt: the traceability footer is a fixed related-document block, not in-prose guidance.

## Traceability footer

- The traceability footer is the only exception to dependency direction and single source: the closing block on a template-bearing document — module-mirroring or mixin — that links it to its related documents; a cross-cutting document carries no footer.
- Each such class fixes its own footer contents in its writing guide, so the family, module, base, leaf, and mixin footers each carry their own lines. A mixin has no Reference twin, so its footer Reference line is always `N/A — software mechanism`.

## Present state only

- Every carrier describes the current state only — no history and no future.
- Remove the entire out-of-state statement, not just its trigger word: delete what the subject once was, what it is not, and what it will become, and state the positive current fact. Never soften "X is no longer Y" into "X is not Y".
