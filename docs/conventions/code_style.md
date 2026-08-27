# Code style

In-code documentation — docstrings, comments, and shape and type annotations — is this document's main subject, together with a few project-specific coding contracts.

A convention document names only a global public base class, a lifecycle method such a base declares, and a name pattern the conventions themselves prescribe — never the class, field, or axis of a functional family or of a leaf module, since editing one module or one family must never force an edit here.

Several rules below are enforced by the static checks under `tests/rules/`. The rule text here stays the authority; a note marks each rule a check already decides.

## Baseline references

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [mypy documentation](https://mypy.readthedocs.io/)
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/)

Follow these public conventions unless a rule here is stricter.

## Characters and notation

[notation_conventions](notation_conventions.md) is the authority for the non-ASCII whitelist, for where a formula may live, and for the full split by string class.

**Effective code is ASCII.** Identifiers and protocol or data string literals use only ASCII. A human-facing string — a log line, a `print`, an exception message — is not effective code and follows the comment whitelist instead.

A docstring or comment carries raw whitelisted unicode, no LaTeX, and no hosted formula. Its inline literals use single backticks; reStructuredText roles and double-backtick literals are not written.

## Interface tiers

Every symbol sits in one of three tiers, and the tier fixes what its docstring owes. The question is who reads the symbol, not where it lives.

- **User API** — what someone using the library calls directly: the network layer objects, the profiling and reporting faces, the configuration entry points. Its docstring states what the signature cannot — the public shape, dtype, and unit contract, the lifecycle precondition a call assumes, a non-obvious side effect, an error the caller must handle — and repeats nothing a parameter name, its type, or its unit suffix already says. The full Google structure is available here.
- **Extension SPI** — what an author of a new implementation writes against: an abstract base, a Protocol, a mixin, a registry face. Its docstring carries the whole extension contract: the behavior the base injects, the members a subclass supplies, the order hooks run in, the ownership and state requirements no type expresses, and how a breach fails. Its length comes from the real contract, never from a template.
- **Implementation detail** — everything else, carrying its meaning in its name, its types, its narrow helpers, a line or two of comment, and a focused test. A private function needs no docstring, nor does an obvious delegating property, and nothing explains that a config argument is a config or that a name ending in a temperature suffix is a temperature.

Knowledge no single symbol owns — a contract two components must both honor — belongs to a [System Design](../system_design/README.md) topic rather than to a longer docstring.

## Docstrings

- Use Google-style docstrings. The recognized sections are the standard Google set (`Args:`, `Returns:`, `Yields:`, `Raises:`), plus `See Also:` in a module docstring. `Examples:` is reserved for explicitly allowlisted user-facing objects, initially `Profiler` and `Reporter`; other APIs keep examples in Guides or API documentation. No other section is written. Caller contract without a standard section stays in the owning docstring body, scientific material and citations go to Reference, cross-component design to System Design, tutorials to Guides, and template content nowhere. The `Examples:` reservation and `See Also:` structure are machine-checked; ordinary prose is not classified from punctuation alone.
- A section covers its items selectively: an entry is written only where it adds information beyond the name and type already visible in the signature, so an `Args:` block documenting two parameters out of five is correct rather than incomplete.
- State a tensor shape where shape is part of the contract; a pure elementwise API may omit it or state the same-shape rule once, and a public tensor container crossing a module boundary chooses whether to state shapes at all.
- A docstring `Shape:` line is written only as the separate final line of an entry inside an `Args:`, `Returns:`, or `Yields:` section, or as the final line of an attribute docstring, and only where that entry describes a tensor, an optional tensor, or a container whose elements are tensors. Its form is `` Shape: `[*inst_shape, item_num]`. `` — an inline code literal closed by a period, occupying its own line. No other position carries the line.
- An override inherits its interface docstring rather than copying it, and a leaf restates no contract it inherits; document only the difference where an override changes contract, shape, side effects, units, or errors.
- An obligation a class places on a subclass or a host is plain normative prose in the class docstring.
- An interface docstring states what the method does, not a directive to whoever implements it: the abstract declaration already carries the obligation, and "a subclass must implement this" stops holding once one has.
- A magic method carries no docstring. State its caller-visible behavior in the class docstring instead. (Machine-checked.)
- A concrete field-defined `Config`, `Policy`, `Snap`, `Dcop`, or `Record` data class normally carries no class docstring. Its class name states its role, its annotations state its structure, and the attribute docstrings immediately following its fields state each field's meaning, unit, shape, and non-obvious constraint. Retain a class docstring only when the class adds behavior, an aggregate invariant, a lifecycle contract, or an implementer obligation that its name, bases, and individual fields do not express. Ruff's `D101` is therefore disabled globally; the need for a class docstring is a semantic decision, not one a suffix-insensitive presence check can enforce.
- A module docstring states the file's responsibility. It may add at most one top-level `See Also:` section listing the documents a maintainer of that file needs — typically the Reference page specifying the model it implements, or the [System Design](../system_design/README.md) topic owning a cross-component contract it takes part in — and a file whose knowledge is its own lists nothing. Its non-empty body is a contiguous list with no internal blank line: every entry is indented four spaces and contains one bare path, with no section name, parenthetical, or prose, and every cited path resolves to a file that exists. A class docstring, a function docstring, and an inline comment carry no document pointer at all. Scientific literature is cited in the owning Reference document; a `References:` bibliography is not an alias for this navigation section. (Machine-checked.)
- A package `__init__.py` docstring is optional: keep one, of one to three sentences, only for a user-facing import boundary, a CLI package whose purpose is not obvious, or a package-level behavior such as import-time registration. A structural or purely re-exporting package has none — its path expresses the structure and `__all__` the public names — and no package docstring carries a member catalog, a design explanation, or a documentation pointer.
- Do not repeat a physical unit in prose when the name already carries a unit suffix; state one only where no suffix exists, the value is normalized or scaled, or the convention is otherwise non-obvious.
- A test module's docstring is one line naming the laws it puts under test; a test function carries one only where its law does not fit in its name, and a fixture value whose choice is not obvious is explained in an inline comment.

## Inline comments

- Use inline comments for local implementation help: non-obvious math, numerical intent, shape, and structural or procedural banners. One to three lines, saying why rather than what.
- A contract spanning files has no comment-sized home: it belongs to the owning symbol's docstring, or to a [System Design](../system_design/README.md) topic.

## Banner comments

An equals-sign banner marks structure at class scope; a dash banner marks procedure inside a body. An equals banner at method indent is a violation, and so is a dash banner at class-body indent. Either form stands on its own line, has one blank line below, and names what it opens. It normally has one blank line above as well. When an equals banner is the first content of a class without a docstring, it directly follows the class header because the formatter removes an intervening blank line. A dash banner following its function or method docstring likewise touches the docstring. A class docstring and the equals banner following it retain the normal blank line. Its three markers on each side and the single padding spaces are fixed; only the text changes. A two-sided comment with the same padding and at least two `=` or `-` markers on each side is a banner candidate and must use one of the exact forms below. One-sided or incompletely padded comments remain prose. An unnamed or box-drawn divider is not written, at any scope. (Machine-checked.)

- **Class scope** — `# === Description ===`, with no prefix word. It has exactly two uses: the declaration groups of a class header and the field partitions of a config or policy dataclass; it never groups methods. The text names the group — ownership or lifecycle for a declaration group, the field theme for a config or policy partition — never a procedural step, and carries no number.
- **Method body, numbered step** — `# --- 2: condense the cell network onto wire nodes ---`, with an integer or dotted hierarchical number such as `2` or `2.1`. The numbering orders the procedure the body itself implements.
- **Method body, unnumbered group** — `# --- Name ---`, where the grouping follows no numbered procedure.

## Package surface

Every package curates its public face in its `__init__.py`: one unannotated literal string list, `__all__ = [...]`, naming exactly what the package exports. Its entries are unique. The face selects rather than mirrors — a name serving only the package's own modules stays off it — and a package with no importable face, a command-line package reached through its scripts, declares none. Dynamic construction, multiple assignments, an annotated declaration, and an alternate container are not written: the list is the one statically readable authority on the package's public names. (Machine-checked.)

- A leading-underscore file contains internal implementation only and supplies no public API. Its containing package subtree may import it, but a package face neither imports from it nor lists it or one of its definitions in `__all__`. A public definition lives in a non-underscore file even though callers still reach that definition through the package face rather than through the concrete file. (Machine-checked.)
- The face does not lift a subpackage's names one level up; it lists the subpackage itself and stops there. Each face therefore stays the sole authority on its own names, and a caller descends one package at a time. The one exception is the root face, which additionally exposes the fabrication, profiling, reporting, and tree-naming entry points drawn from the shared-kernel subpackage, so the whole top-level user API sits behind one import.
- Import form follows ownership and package boundaries. A module reaches a sibling file in its own package by a single-dot relative file import, and a direct child package by a single-dot import from that child's package face. A descendant reaches a `.py` file directly owned by any strict ancestor directory through that file's absolute module path, never through the ancestor's `__init__.py` face. Every other relationship — including another subpackage under a parent directory — is cross-package and uses the target package's absolute `__all__` face, never one of its concrete files. A multi-dot relative import is not written. (Machine-checked with Ruff and the rules suite.)
- The routing rule applies equally when library modules, command-line tools, or validation scripts import NeuroX code. A command-line-only subtree under `neurox/tools` declares no package face of its own; its imports still follow the same ownership cases, including absolute access from a descendant to a private file directly owned by the strict ancestor `neurox.tools` directory. Validation campaigns are callers rather than library packages: they obey NeuroX package faces but do not acquire their own `__init__.py` or `__all__` hierarchy.

## Top-level dependency direction

Top-level source directories carry coarse roles, not a total ordering of every internal package. `common` depends only on itself; `primitive` may additionally depend on `common`; `architecture` may depend on both core roles; `works` holds concrete design extensions that may combine all core roles; and `tools` is a terminal consumer of the full stack. A role may depend on itself. Core library code never imports `works` or `tools`, and no library role imports `tools`. This ruling deliberately says nothing about a complete ordering among nested device, circuit, macro, mapping, or engine packages: their ownership is expressed by their package boundaries and import faces. (Machine-checked.)

## Class header

Order a class header: docstring, class variable assignments, grouped state declarations, then `__init__`. In a class that defines `__init__`, it precedes every other `def`; a class that defines none — a mixin, or a config or policy dataclass — is unaffected.

## Class variables

- A class variable defines and assigns a real value shared by every instance; a state declaration describes per-instance state and assigns at most an absent-state default.
- A class variable assignment carries no banner.
- A class reporting no silicon area of its own — one whose area and leakage are counted at its owner, or a pure container owning no silicon — declares the profile-target class variable false.
- A declaration that carries a class-scope default — an optional programmed tensor defaulting to absent — is a state declaration, not a class variable, and stays inside its lifecycle group.

## Class state declarations

The class header is the object's state manifest for the human reader; a declaration is kept even where a direct assignment would let mypy infer the attribute.

- Exactly three kinds are declared: buffers registered in `__init__`, tensors produced by `fabricate()`, and tensors produced by `program(...)`.
- A child object or submodule is not declared — one visible assignment in `__init__` and the module tree already expose it — and neither is compact scalar metadata bound there. A subclass may repeat an inherited child declaration solely to narrow its static type; that declaration introduces no new state.
- An attribute a base class requires of its subclass is declared as an abstract property on the base, per §Property vs method.
- A config, policy, or Protocol field, or a field of a public tensor container, stays explicit because its declaration defines that data structure; it is no state declaration, so the grouping, typing, and shape rules below do not reach it.
- Register each buffer through `ModuleBase._register_nonpersistent_buffer` with an explicit literal name; never hide buffer creation behind a loop or `setattr`.
- Only a registered buffer is called a buffer; what `fabricate()` or `program(...)` produces is state.

Group declarations by lifecycle phase, under these names and in this order, omitting any group the class does not have:

| Group | Holds |
|---|---|
| Functional buffers | Registered tensors execution reads: lookup tables, ratio vectors, indices and masks, tensor-valued constants, and 0-D device/dtype or expansion seeds |
| Nominal buffers | The registered fabrication sources `fabricate()` consumes |
| Fabricated state | What `fabricate()` produces |
| Programmed state | What `program(...)` produces |

- A banner opens a group where the grouping helps the reader; a group holding a single member needs none.
- Where the nominal and fabricated groups pair one-to-one, both list their members in the same order.
- Declare the type an attribute holds once its lifecycle phase has run. Use an optional type only where absence is itself a supported state, such as a programmed tensor a caller can clear; not-yet-fabricated is a caller contract violation, not a supported state.
- Every declaration in these groups carries a trailing shape annotation, per §Shape annotations.

## Construction

- Keep `__init__` focused on validation, binding compact scalar metadata, and ordering construction phases.
- Move a substantial child-object construction block to a narrowly named `_init_*_children(...)` helper.
- Move a substantial nominal-buffer registration block to `_register_fabrication_buffers(...)`. Fixed functional LUTs and other short functional-buffer registrations may remain inline.
- The class-scope entry for fabricated or programmed state declares the attribute without materializing it: `fabricate()` and `program(...)` create ordinary tensor attributes after module device migration, and `__init__` leaves no placeholder.
- `fabricate()` and `program(...)` materialize the already-declared shape and nothing more: broadcast or expand a nominal buffer to the instance shape, draw at the declared shape, combine elementwise. A step that transforms axes in any of the ways §Shape annotations lists moves into a helper, which annotates normally.
- Do not split short constructors mechanically; a helper exposes a real construction phase, not a few moved assignments.

## Shape annotations

The `Shape: ` label is written in two forms: the in-code `# Shape:` comment, fixed here, and the final line of a docstring field entry, fixed in §Docstrings. Both fill their line entirely and share the grammar inside the brackets; only the rendering differs, the in-code form carrying no terminating period. A shape annotation is the in-code form. A public tensor container crossing a module boundary states any shape in the docstring, where visibility is widest; private state states it in code, and no caller-visible shape contract hides in an inline comment.

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

- Write a standalone annotation only at a real boundary or a non-obvious transform: a cross-module call whose result shape the reader cannot infer, or a step that inserts, removes, merges, splits, permutes, reduces, broadcast-aligns, chunks, or reassembles axes. Skip pure elementwise code and code operating at a known contract shape — `inst_shape` together with the private axes the module owns, a shape a class-header declaration states, or an explicit per-call `shape` — unless it delegates to a helper that transforms layout. Never annotate every tensor line.
- A standalone annotation states the input and the output of the one statement it annotates, using at most one arrow. Write a longer chain only where an intermediate shape is both important to reading the line and unrecoverable from it, such as the result of an index-notation contraction; an intermediate the line already spells out fails that test, so collapse the chain to its endpoints. Otherwise a second arrow means the statement does more than one thing, and the statement is split.
- An annotation contains only dimension names, integer literals, `*`, `...`, `->`, `<name>=1` broadcast placeholders, and arithmetic expressing a derived extent over names, literals, or whole bracket groups — `**`, `+`, `-`, and multiplication written infix, `*` elementwise or `@` matrix, with a leading `*` still the unpack prefix; a placeholder names the semantic axis it aligns with. State only the shape; a layout or ordering note goes in an ordinary comment, never inside the annotation.
- Three notations are available and one test picks among them. An axis known concretely is named; an axis group the code owns is unpacked with the `*` prefix when its boundary is load-bearing and its name resolves for the reader; everything else is `...`, standing for any number of axes, possibly none, whose identity the annotation does not care about. Two ellipses may coexist in one shape — `[..., name, ...]` — each standing for such a group independently, and an empty `[]` states a 0-D tensor.
- A boundary is load-bearing when something downstream depends on where it falls — a contract that slices the shape by rank, or a rank a consumer declares and the layout is validated against. `...` asserts the opposite, so a caller prefix the code merely broadcasts over takes the ellipsis, while a group a downstream contract slices stays a named slot even where the code cannot see inside it.
- A name resolves against the reader the shape addresses. An in-code annotation addresses someone inside the body, so every name in scope there resolves; a docstring addresses the caller, which resolves only a public attribute or property, a public config field name, and an axis name the documentation defines, so a body-local name in a docstring shape is a defect.
- A `float`, `int`, or `bool` cast, an `.item()`, or a `.tolist()` wrapping a multi-step tensor transform takes its own line: compute the tensor on one line, annotated as the rules above call for, then convert on the next. A conversion whose tensor side is a single step, such as a cast around one reduction, is the ordinary idiom and is never split.

## Dynamic-energy axes

Every energy tensor handed to the profiler carries one repo-wide axis layout, `[*caller_leading, ...]`: the measuring caller's own batch or time prefix first, one position per independent unit operation, then the emitter's internal structure — a serialized round, a digit, a phase, an output slot, a fabrication instance. The caller declares its own rank once; every axis past that block is summed, no kind of axis distinguished from another, so an emitter declares nothing about its axes and orders them however its math produces them.

An emitter owes two clauses on the tensor it hands over. **It carries the caller's leading dims**: never pre-reduce them, never emit below the declared rank, and bill from reassembled full-shape state when the work runs in chunks, a chunk axis having ravelled the caller's block into one axis that cannot express the layout. **It carries them at their true extents**: a size-one stand-in for a real caller extent is a contract violation, not a broadcast request — nothing expands it, so the event is summed as the single unit operation it claims to be and under-counts that emitter by the caller batch's product.

Billing runs under no-grad: build the energy tensor inside `torch.no_grad()` or an equivalent guard, so no billing arithmetic enters the autograd graph. The detach applied on submission is a backstop against a missed guard, not the mechanism.

At an emission site the shape annotation writes the caller block as a named group under the `*` prefix, a downstream contract slicing the tensor by that rank, and everything after it as `...`. Constant-per-element billing builds a 0-dim tensor holding the per-op constant and expands it onto the billed layout, so no full constant energy tensor is ever materialized; that constant fixes the energy dtype, the billed layout being typically an integer code or a reduced-precision signal.

## Signatures and types

- Annotate every parameter and the return type in a function or method signature. §Class state declarations fixes the type an attribute declaration carries.
- A default is legal only where it is the identity value — the no-op, the neutral element, or the absent state — so that omitting the argument and passing the default are the same call. A parameter whose value picks behavior, selects a policy, or states a physical quantity carries no default and is named at every call site.
- Annotate with `object` only an operation that genuinely never inspects the value. Anything that reads, converts, or dispatches on it takes `Any`, or better the abstract base it actually requires.
- Treat mypy as a helper, not a gate. When a false positive comes from an external library or a pattern mypy cannot express — a TypeVar not re-bound after an `isinstance` narrowing, a `fields()` or `replace()` call needing a `DataclassInstance` — leave the error unsuppressed. Never write `# type: ignore`.
- In reusable core code, fix a base-class-related narrowing in the generic contract rather than reach for `cast`. A final concrete implementation under `neurox.works` may cast at an override boundary when its own construction and dispatch uniquely fix the runtime subtype that the base signature erased. Keep that cast local to the boundary; never use this exception for registry results, external input, optional values, tensor properties, or unrelated types.
- Never add a meaningless runtime conversion only to satisfy typing.
- `ModuleBase` binds only `ConfigT` and `PolicyT`, the two roles every physical module owns. A family base adds a type parameter for each further associated type that appears in its interface; it does not burden unrelated module families with dummy `SnapT` or `DcopT` parameters.
- A reusable implementation layer remains generic over every associated type its family is designed to vary, and each concrete leaf binds that tuple once. Do not extract a synthetic shared base solely so one final work-specific implementation can extend an otherwise complete concrete model; that edge may inherit the model and recover an erased subtype under the local-cast rule above. If independent descendants make the variation recurring, promote it into a real generic implementation layer.
- A registry factory returns its family abstraction: runtime dispatch proves which registered class was selected, but the static type does not pretend to recover that class's complete generic specialization. Keep any erased associated type at this construction boundary. Code that depends on a concrete member's extended interface constructs that member through a typed path instead of casting a registry result; downstream helpers preserve the exact type they receive and never widen it to `Any`.

## Config, policy, and cross-module data classes

- Declare config and policy descendants as ordinary classes inheriting `ConfigBase` or `PolicyBase`. The roots apply `dataclass(frozen=True, kw_only=True)` to every descendant; never repeat `@dataclass` or write `__init__` on one.
- A formal snapshot, DC operating point, or recorder row inherits `SnapBase`, `DcopBase`, or `RecordBase`. These role roots derive from `TensorDataClassBase`, which supplies `dataclass(eq=False, frozen=True, kw_only=True)`; `SnapBase` also supplies `TensorGroupMixin`, because every snap's tensor fields share one per-call layout and transform together. Other tensor data classes opt into `TensorDataClassBase` when they need the same identity-equality and construction semantics; carrying a tensor alone does not impose that inheritance hierarchy. A class in any of these hierarchies never repeats `@dataclass`.
- Declare every field in a base-managed descendant with an annotation and no value. The bases refuse any field name assigned in the class body.
- Which config a parameter belongs on follows the variability criterion in [module_parameter §Config layering](module_parameter.md#config-layering).
- A dataclass field — however the transform is applied — an enum member, a named-tuple field, or a public instance attribute assigned in `__init__` carries its documentation as a string literal immediately below the declaration, written only where the member needs more than its name and type already state.
- A field partition carries a class-scope banner. Partition names are per-config and free-form.
- Put config- and policy-domain checks in `validate()`. Both roots invoke the most-derived implementation after construction, so a descendant never declares `__post_init__` or calls `validate()` itself.
- Preserve inherited validation. A descendant extending a parent with constraints calls `super().validate()` before checking its own fields.
- Validate independent fields with explicit statements that name the attribute and its error label directly. Do not loop over field-name strings, use `getattr()`, or combine fields merely to shorten adjacent checks. A loop remains appropriate when one field is itself a variable-length collection whose elements must all be checked, or when a relationship is defined elementwise across variable-length collections.
- Keep single-use validation groups inline in `validate()`, separated by an unnumbered method-body banner where that improves scanning. Extract a `_validate_*` helper only when the same check is genuinely reused or the helper implements a substantial standalone algorithm, never merely to move adjacent checks elsewhere.

## Guards and exceptions

A guard is split by cause, and its exception class follows what the guard checks.

- **One domain, one guard.** Predicates that jointly define a single valid domain for one value — finite and inside its range, or one shape contract — form one check that raises once, with a combined message stating the whole domain.
- **Distinct causes, separate guards.** Predicates that fail for different reasons are separate checks, each raising its own precise message: a wrong type against a bad value, a container against its elements, one variable against another, an absent key against a key holding the wrong type. An element-level raise names the offending index.
- **Class follows the check.** A wrong type raises `TypeError`; a lookup miss, an absent key or segment, raises `KeyError`; an illegal key set, a value outside its domain, or invalid content raises `ValueError`; a fatal condition in a command-line entry script raises `SystemExit`.

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
- **A block that only occupies area is a seat, not a subclass.** Two degeneracies collapse below a subclass. Pure linear current combining — scaling, summing, or differencing branch currents — is Kirchhoff's current law, so it is tensor arithmetic in the composing module, not a class at all. A real block whose only footprint is static area and leakage plus a data-independent per-op energy is a static seat: a shared unmodeled-PPA block plus a per-op constant billed by the composite. The seat is a no-noise expedient — a stateful non-ideality, an offset sampled and held or a mismatch drawn per instance, restores the block to a class.

## Compile safety

Library code is written to be traceable, so a caller's `torch.compile` gets a graph and the library's own compiled leaf traces cleanly. These invariants hold everywhere but at the declared eager boundaries, which [compile boundary](../system_design/compile.md) maps.

- **No host sync.** `.item()`, `.tolist()`, an `int`, `float`, or `bool` cast of a tensor, `.cpu()`, `.numpy()`, `.to("cpu")`, and printing a tensor all force a sync dynamo cannot trace. `tensor.shape[i]`, `.size(i)`, and `.ndim` return a Python `int` without a sync and are safe, as is a bare `.detach()`.
- **No Python-state mutation.** No attribute assignment on `self`, no mutation of an externally reachable container, no buffer or parameter registration, and no training-mode toggle on a traced path. Accumulating into a list is legal only inside an eager island.
- **No branching on tensor values.** A data-dependent branch forces a graph break; express the choice with masking, `torch.where`, or indexing. A branch on the Python type of an argument or on a Python `bool` resolves at trace time and is safe; it, and any unavoidable value-dependent branch, carries an inline comment saying why.
- **No rebuilding of fixed tensor sources on the run path.** A config- or design-derived constant is registered at construction and migrates with the module; a runtime-derived tensor is built with `*_like` or `new_*` off a semantically related input, never off an unrelated tensor borrowed as a dtype or device anchor.
- **Grad guards trace.** `torch.no_grad()` and `torch.enable_grad()` are safe, though the latter makes dynamo more conservative.

Library code tolerates graph breaks, so `fullgraph` stays off; `fullgraph=True` is a test-side tool for catching an accidental sync. `@torch_compiler_disable` is no license to sync — disabling another function to silence a trace error hides a real violation, so fix the control flow or the state mutation instead.

## Device placement

The library never selects a device. An entry point accepting `"auto"` resolves it to CUDA when CUDA is available and to CPU otherwise, reading no other signal — not free memory, not utilization, not a device inventory. Which card a run lands on is the caller's decision, made through the environment.
