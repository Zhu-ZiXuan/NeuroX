# Organizing principles

Documentation is carried in two media: markdown files and in-code text (docstrings, comments, shape and type annotations).

## What each carrier holds

| Carrier | Holds |
|---|---|
| `docs/reference/**.md` | Researcher-voice scientific, mathematical, circuit, architecture, and algorithm principles and design; the golden truth code translates; independent of the concrete implementation |
| `docs/internals/**.md` | Engineer-voice Python system design; decisions, trade-offs, and cross-file contracts not readable from code |
| `docs/contributing/**.md` | How to write documentation and code |
| docstring | Caller-facing API semantics of the symbol |
| inline comment | Local implementation intent |
| shape annotation | Semantic tensor-shape transitions at the point of code; its format is in [code_style](code_style.md) |
| step comment | Procedural phase boundaries aligned to a numbered procedure in Reference or Internals; its format is in [code_style](code_style.md) |

When a statement could fit two carriers: survives a code rewrite → Reference; describes the program → Internals; caller-facing → docstring; local intent → comment.

The same split governs tensor shape: Reference owns scientific or mathematical tensor meaning; Internals owns implementation layout, broadcast, batching, and reshape invariants.

## Module and cross-cutting documents

Reference and Internals mirror the core library to directory granularity, in two kinds of document:

- **Module document** — mirrors a single code module, lives in a subdirectory, and follows the module convention: the prescribed section structure and the traceability footer.
- **Cross-cutting document** — spans modules, sits at the Reference or Internals root, carries neither the prescribed sections nor a footer, and is organized freely around its own content.

Reference is a knowledge-subset mirror: a module earns a Reference document only when it carries knowledge content — mathematics, physics, a device or circuit model, an architecture, or an algorithm. A pure-programming module has no Reference document and is documented only in Internals.

A rule or principle document states these criteria in general terms, not by enumerating specific subdirectory names.

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
- Name an area briefly and route the reader through README and recipes; link a specific shared contract only when a sentence depends on it.

## Links and navigation

- No document outside README carries a topic-link index or navigation menu in its header; navigation is README's alone, and every other document opens directly on its own subject.
- A reader follows every in-prose link, and each one recursively. Place an in-prose guiding link only when its target genuinely holds information both relevant to the sentence and correct there; otherwise omit it.
- Footer links are exempt: the traceability footer is a fixed related-document block, not in-prose guidance.

## Traceability footer

- The traceability footer is the only exception to dependency direction and single source: the closing block that links a module document to its related documents.
- Its contents are defined in [writing_reference_docs](../contributing/writing_reference_docs.md) and [writing_internals_docs](../contributing/writing_internals_docs.md).

## Present state only

- Every carrier describes the current state only.
- An ADR is the only place for historical narrative, such as what a module was once called or why an earlier approach was replaced.
