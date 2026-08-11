# Code style

In-code documentation — docstrings, comments, and shape and type annotations — is the main subject of how code files are written, plus a few project-specific coding contracts.

A convention document may name a global public base class, a lifecycle method such a base declares, and a name pattern the conventions themselves prescribe. It never names the class, field, or axis of a functional family or of a leaf module: editing one module or one family must never force an edit to a convention document.

## Baseline references

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [mypy documentation](https://mypy.readthedocs.io/)
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/)

Follow these public conventions unless a rule here is stricter.

## Characters and notation

[notation_conventions](notation_conventions.md) is the authority for the non-ASCII whitelist and for where a formula may live.

**Effective code is ASCII.** Identifiers and protocol or data string literals use only ASCII. A human-facing string — a log line, a `print`, an exception message — is not effective code; it follows the comment whitelist, not this rule. notation_conventions §Notation by string class gives the full split.

A docstring or comment carries raw whitelisted unicode, no LaTeX, and no hosted formula, per [notation_conventions](notation_conventions.md).

## Docstrings

- Use Google-style docstrings.
- Write for the caller of the symbol: public semantics, and tensor shapes when shape is part of the public contract. A pure elementwise API may omit shape or state the same-shape rule once.
- A docstring `Shape:` line is written only as the separate final line of an entry inside an `Attributes:`, `Args:`, `Parameters:`, `Returns:`, or `Yields:` section, and only where that entry describes an actual tensor, an optional tensor, or a container whose elements are tensors. Its form is ````Shape: ``[*inst_shape, item_num]``.```` — a reStructuredText inline literal closed by a period, occupying its own line. Whether a public tensor payload that crosses a module boundary states shapes at all is its own call; when it does, it uses this form.
- No other position carries the line: never trailing a free prose paragraph, never at the end of a module-level docstring, never inside a `Raises:`, `Note:`, `Example:`, or `See Also:` section, and never on an entry whose type is not a tensor.
- Write the complete callable interface on the abstract method, mixin method, or Protocol method. An unchanged override inherits it instead of copying it; document only the difference when an override changes contract, shape, side effects, units, or errors.
- A base or mixin class docstring contains only a short responsibility statement and requirements imposed on subclasses or hosts. General guidance, design rationale, lifecycle, ownership, and implementation details belong in the relevant Conventions or Internals document.
- An interface docstring states what the method does, not a directive to whoever implements it — "a subclass must implement this" stops holding once one has. The obligation to implement belongs in the class docstring, the not-yet-implemented fact in `raise NotImplementedError`, and the rationale in Internals.
- A lifecycle magic method (`__post_init__`, `__init_subclass__`) carries no docstring — a caller never invokes it directly, so the docstring would go unread. State the behavior it drives in the class docstring instead.
- A regular implementation module's docstring states the file's responsibility. When a matching Reference or Internals document exists, it must include a `See also:` entry pointing to that document.
- A package `__init__.py` docstring is optional. Keep one only for a user-facing import boundary, a CLI package whose purpose is not obvious, or a package-level behavior such as import-time registration. State only that package responsibility or public behavior in one to three sentences.
- An internal structural package or an `__init__.py` that only re-exports names has no docstring. Its path expresses the structure and `__all__` expresses the public names; a generic restatement of either adds no information.
- A package docstring carries no member catalog, design explanation, `See also:` entry, or other documentation link. Documentation discovery belongs to the documentation navigation rather than the import surface.
- In a regular implementation module, a pointer from code to a document appears only in the module docstring's `See also:` entry and is bare: the document path alone, with no section name, parenthetical, or prose. A function docstring and an inline comment carry no such pointer. A numbered banner comment aligned to a numbered procedure is not a document pointer and is exempt.
- Do not repeat a physical unit in prose when the name already carries a unit suffix. State a unit only when no suffix exists, the value is normalized or scaled, or the convention is otherwise non-obvious.

## Inline comments

- Use inline comments for local implementation help: non-obvious math, numerical intent, shape, and structural or procedural banners.
- Design rationale, lifecycle, ownership, and cross-file contracts belong in Internals.

## Banner comments

An equals-sign banner marks structure at class scope; a dash banner marks procedure inside a body. An equals banner at method indent is a violation, and so is a dash banner at class-body indent. Either form stands on its own line with one blank line above and below. A banner names what it opens; an unnamed or box-drawn divider is not written, at any scope.

- **Class scope** — `# === Description ===`, with no prefix word. It has exactly two uses: the declaration groups of a class header and the field partitions of a config or policy dataclass; it never groups methods. The text names the group — ownership or lifecycle for a declaration group, the field theme for a config or policy partition — never a procedural step, and carries no step number.
- **Method body, numbered step** — `# --- 2: condense the cell network onto wire nodes ---`, with an integer or dotted hierarchical number such as `2` or `2.1`. Align each number and name with the ordered procedure in the module's Reference or Internals document.
- **Method body, unnumbered group** — `# --- Name ---`, where the grouping follows no numbered procedure.

## Class header

Order a class header: docstring, class variable assignments, grouped state declarations, then `__init__`. In a class that defines `__init__`, it precedes every other `def`; a class that defines none — a mixin, or a config or policy dataclass — is unaffected by the rule.

## Class variables

- A class variable defines and assigns a real value shared by every instance; a state declaration describes per-instance state and assigns at most an absent-state default.
- A class variable assignment carries no banner.
- A class that does not report silicon area of its own — a primitive whose area and leakage are already counted at its owner, or a pure container that owns no silicon — declares the profile-target class variable false.
- A declaration that carries a class-scope default — an optional programmed tensor defaulting to absent — is a state declaration, not a class variable, and stays inside its lifecycle group.

## Class state declarations

The class header is the object's state manifest, written for the human reader; a declaration is kept even where a direct assignment would let mypy infer the attribute.

- Exactly three kinds are declared: buffers registered in `__init__`, tensors produced by `fabricate()`, and tensors produced by `program(...)`.
- A child object or submodule is not declared: it has one visible assignment in `__init__`, and the module tree already exposes it. Compact scalar metadata bound in `__init__` is not declared either.
- An attribute a base class requires of its subclass is declared as an abstract property on the base, per §Property vs method.
- A config, policy, or Protocol field, or a field of a public tensor payload that crosses a module boundary, stays explicit, because its declaration defines that data structure; it is not a state declaration, so the grouping, typing, and shape rules below do not apply to it.
- Register each buffer with an explicit literal name; never hide buffer creation behind a loop or `setattr`.
- Only a registered buffer is called a buffer; what `fabricate()` or `program(...)` produces is state.

Group declarations by lifecycle phase, under these names and in this order, omitting any group the class does not have:

| Group | Holds |
|---|---|
| Functional buffers | Registered buffers the forward math reads: lookup tables, ratio vectors, index and mapping masks, bias constants |
| Circuit constant buffers | Registered buffers holding electrical and timing constants |
| Nominal buffers | The registered fabrication sources `fabricate()` consumes |
| Fabricated state | What `fabricate()` produces |
| Programmed state | What `program(...)` produces |
| Runtime buffers | Registered buffers the forward mutates in place, whose lifecycle is training or calibration rather than fabrication or programming |

- Every group carries its own banner, including a group holding a single member.
- Where the nominal and fabricated groups pair one-to-one, both list their members in the same order.
- Declare the type an attribute holds once its lifecycle phase has run. Use an optional type only where absence is itself a supported state, such as a programmed tensor a caller can clear; not-yet-fabricated is a caller contract violation, not a supported state.
- Every declaration in these groups carries a trailing shape annotation, per §Shape annotations.

## Construction

- Keep `__init__` focused on validation, binding compact scalar metadata, and ordering construction phases.
- Move a substantial child-object construction block to a narrowly named `_init_*_children(...)` helper.
- Move a substantial nominal-buffer registration block to `_register_fabrication_buffers(...)`. Fixed functional LUTs and other short functional-buffer registrations may remain inline.
- The class-scope entry for fabricated or programmed state declares the attribute without materializing it; `fabricate()` and `program(...)` create ordinary tensor attributes after module device migration, and `__init__` materializes no placeholder for them.
- `fabricate()` and `program(...)` materialize the already-declared shape and nothing more: broadcast or expand a nominal buffer to the instance shape, draw at the declared shape, combine elementwise. A step that transforms axes in any of the ways §Shape annotations lists moves into a helper, which annotates normally.
- Do not split short constructors mechanically; a helper must expose a real construction phase rather than merely move a few assignments.

## Shape annotations

The `Shape: ` label is written in two forms: the in-code `# Shape:` comment, fixed here, and the final line of a docstring field entry, fixed in §Docstrings. Both fill their comment or their docstring line entirely and use the same grammar inside the brackets; only the rendering differs, and the in-code form carries no terminating period. A shape annotation is the in-code form; the shape a docstring states is not one. A public tensor payload that crosses a module boundary states its shape in the docstring, where visibility is widest, and private state states it in code. The public shape contract itself is stated by the owning Reference or Internals document or by the public docstring.

- Do not hide a caller-visible shape contract in an inline comment.
- On a class-header declaration the annotation trails the declaration and states the attribute's terminal shape:

  ```python
  _name: Tensor  # Shape: [*inst_shape, item_num]
  ```

  Exactly two spaces precede the hash, no line is padded to align with another, and the comment carries no semantic note.
- Anywhere else the annotation is a standalone comment line above the statement and states a shape transition:

  ```python
  # Shape: [new]
  # Shape: [old] -> [new]
  ```

- A standalone annotation states the input and the output of the one statement it annotates, using at most one arrow. A longer chain is written only where an intermediate shape is both important to reading the line and unrecoverable from the line itself — the result shape of a call the reader cannot infer, such as an index-notation contraction or a cross-module call. An intermediate the line already spells out fails that test: replaying the visible sequence of calls restates the code, so collapse the chain to its endpoints.
- Add a standalone annotation where a function or helper inserts, removes, merges, splits, permutes, reduces, broadcast-aligns, chunks, or reassembles axes. Skip pure elementwise code and code that operates at a known contract shape — `inst_shape` together with whatever private axes the module owns, a shape stated by a class-header declaration, or an explicit per-call `shape` — unless it delegates to a helper that transforms layout. Annotate only where the shape changes or becomes non-obvious, not every tensor line.
- An annotation contains only dimension names, integer literals, `*`, `...`, `->`, `<name>=1` broadcast placeholders, and arithmetic expressing a derived extent over names, literals, or whole bracket groups — `**`, `+`, `-`, and multiplication written infix, `*` elementwise or `@` matrix, with a leading `*` still the unpack prefix; a placeholder must name the semantic axis it aligns with. State only the shape; put any layout or ordering note in an ordinary comment, never inside the annotation.
- Two notations carry a fixed meaning. `...` stands for any number of axes, possibly none, whose identity the annotation does not care about, and two of them may coexist in one shape — `[..., name, ...]` — each standing for such a group independently; an empty `[]` states a 0-D tensor. An axis the annotation does care about is named, and a named axis group the code owns is unpacked with the `*` prefix.
- Three notations are available and one test picks among them. Axes known concretely are named one by one. An axis group is written with the `*` prefix when its boundary is load-bearing and its name resolves for the reader. Everything else is `...`.
- A boundary is load-bearing when something downstream depends on where it falls — a contract that slices the shape by rank, or a rank a consumer declares and the layout is validated against. `...` asserts the opposite, so a caller prefix the code merely broadcasts over takes the ellipsis, while a group a downstream contract slices stays a named slot even where the code cannot see inside it.
- A name resolves against the reader the shape addresses. An in-code annotation addresses someone standing inside the body, so every name in scope there resolves. A docstring addresses the caller, which resolves only a public attribute or property, a public config field name, and an axis name the documentation defines; a body-local name in a docstring shape is a defect.
- An annotation that carries more than one arrow without meeting the longer-chain test above signals a statement doing more than one thing, and that statement is split. In particular, a tensor-to-Python-scalar conversion — a `float`, `int`, or `bool` cast, `.item()`, `.tolist()` — that wraps a multi-step tensor transform takes its own line: compute the tensor on one line, annotated with a single arrow where the rules above call for one, then convert on the following line. A conversion whose tensor side is a single step, such as a cast around one reduction, is the ordinary idiom and is never split.

## Profile payload axes

Every dynamic-energy payload handed to the profiler carries one repo-wide axis layout, `[*caller_leading, ...]`, under one reduction rule: keep the caller's leading dims, sum every axis past them.

- **Caller leading** — the measuring caller's own batch or time prefix, one position per independent unit operation. Only the caller knows its rank, so the caller declares that rank to the profiler once and every payload keeps the block: an event element is the cost of one unit operation, never a figure already collapsed across the caller's batch.
- **Everything after it** — the emitter's own internal structure: a serialized round, a digit, a phase, an output slot, a fabrication instance. All of it is summed, with no distinction drawn between one kind of axis and another, so an emitter declares nothing about its own axes. It lays the caller's block out first and puts its own axes after it, in whatever order its math produces.

The asymmetry is deliberate. The caller is the only party that can know its own rank, while a declaration from the emitter would be unverifiable: folding a fabrication axis and folding a work axis are both plain summation, so no test, gate, or calibration could separate a right declaration from a wrong one.

What the emitter owes instead is a two-clause contract on the payload it hands over:

1. **It carries the caller's leading dims.** Never pre-reduce them, and never emit at a rank below the declared one. An emitter that works in chunks therefore bills from reassembled full-shape state, because a chunk axis has ravelled the caller's block into one axis that cannot express the layout.
2. **It carries them at their true extents.** A size-one stand-in for a real caller extent is a contract violation, not a broadcast request: nothing expands it, so the event is summed as the single unit operation it claims to be and under-counts that emitter by the caller batch's product.

An emission site's shape annotation spells the layout out as written above. The caller block is a named group under the `*` prefix, because a downstream contract slices the payload by that rank; everything after it is `...`, because nothing depends on where its internal boundaries fall. §Shape annotations gives the general test both choices follow from.

One further rule follows from the layout. Constant-per-element billing builds the payload as a 0-dim tensor holding the per-op constant, expanded onto the billed layout: the expanded view holds no storage and the collector's reduction over its stride-0 axes builds only the caller block, so never materialize a full constant payload. The constant fixes the energy dtype, since the billed layout is typically an integer code or a reduced-precision signal.

That summation is also the layout's price: the collector holds no per-instance resolution, and no way of shaping a payload gives it one.

The collector-side contract and its report surface are in [profiler](../internals/common/profiler.md).

## Type annotations

- Annotate every parameter and the return type in a function or method signature. §Class state declarations fixes the type an attribute declaration carries.
- Treat mypy as the baseline. When a false positive comes from an external library or a pattern mypy cannot express — a TypeVar not re-bound after an `isinstance` narrowing, a `fields()` or `replace()` call needing a `DataclassInstance` — leave the error unsuppressed; the project treats mypy as a helper. Never write `# type: ignore`.
- For base-class-related narrowing, fix the generic rather than reach for `cast`.
- Never add a meaningless runtime conversion only to satisfy typing.

## Config and policy dataclasses

- Declare config and policy descendants as ordinary classes inheriting `ConfigBase` or `PolicyBase`. The roots automatically apply `dataclass(frozen=True, kw_only=True)` to every descendant; never repeat `@dataclass` or write `__init__` on one.
- Declare every field with an annotation and document it under `Attributes:` in the class docstring. Follow the explicit-value and parameter-ownership rules in [config and policy](../internals/config_and_policy.md).
- A field partition carries a class-scope banner. Partition names are per-config and free-form.
- Put config- and policy-domain checks in `validate()`. Both roots invoke the most-derived implementation after construction, so a descendant never declares `__post_init__` or calls `validate()` itself.
- Preserve inherited validation. A descendant extending a parent with constraints calls `super().validate()` before checking its own fields.
- Keep single-use validation groups inline in `validate()`. Separate substantial groups with an unnumbered method-body banner when that improves scanning; do not create `_validate_*` methods merely to move adjacent checks elsewhere.
- Extract a validation helper only when the same check is genuinely reused or the helper implements a substantial standalone algorithm.
- Treat freezing as shallow. Config and policy fields use immutable value types unless mutation is explicitly part of the field contract.

## Property vs method

Use `@property` only for a value fixed by construction, computed with at most a cheap, side-effect-free expression over init-fixed inputs (e.g. a shape product). The allowed cases are:

- Config-field accessor: exposes a field of the owned config.
- Transparent delegation: forwards to an attribute of an owned object.
- Abstract base or Protocol contract: declares an attribute-shaped interface point.
- One-step arithmetic over init-fixed state: a single cheap expression.

Anything that touches a runtime tensor, performs real computation, or has a side effect is a method, as is anything that depends on call arguments, runtime mode, or mutation.

## Subclass vs configuration

A class family splits along structural diversity, not parameter diversity.

- **Topology delta earns a subclass.** A member that conducts, bills, or is shaped differently from the base — an extra stage, a branch the base does not draw or account — is a subclass. It flips whatever reporting flag it needs on, adds its own config fields and PPA seat, and implements the base's overridable hook.
- **Coefficient delta stays a config field.** The same topology with different ratios, ranges, or window values is a config field on the shared base, never a subclass. Implementation diversity that a config can hold is coefficient diversity.
- **A base hook is the escape hatch.** The base carries an overridable hook that returns the neutral value and leaves the base a non-reporter, so a coefficient-only member reuses the base unchanged while a structural subclass overrides the hook and turns its flag on.
- **A block that only occupies area is a seat, not a subclass.** Two degeneracies collapse below a subclass. Pure linear current combining — scaling, summing, or differencing branch currents — is Kirchhoff's current law, so it is tensor arithmetic in the composing module, not a class at all. A real block whose only footprint is static area and leakage plus a data-independent per-op energy is a static seat: a shared unmodeled-PPA block plus a per-op constant billed by the composite, not a subclass of its own. This seat is a no-noise expedient — introducing a stateful non-ideality (an offset that must be sampled and held, a mismatch drawn per instance) makes the block stateful again and restores it to a class.

Test: if two members differ only in numbers a config can carry, they are one class; if one conducts, bills, or is shaped differently, they are two.

## Control flow

Do not branch on tensor values. A data-dependent branch forces a graph break under the caller's `torch.compile`; express the choice with tensor operations such as masking, `torch.where`, or indexing. When a value-dependent branch is unavoidable, explain the reason in an inline comment.
