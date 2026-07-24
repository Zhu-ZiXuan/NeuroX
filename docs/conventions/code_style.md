# Code style

In-code documentation — docstrings, comments, and shape and type annotations — is the main subject of how code files are written, plus a few project-specific coding contracts.

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
- Write the complete callable interface on the abstract method, mixin method, or Protocol method. An unchanged override inherits it instead of copying it; document only the difference when an override changes contract, shape, side effects, units, or errors.
- A base or mixin class docstring contains only a short responsibility statement and requirements imposed on subclasses or hosts. General guidance, design rationale, lifecycle, ownership, and implementation details belong in the relevant Conventions or Internals document.
- An interface docstring states what the method does, not a directive to whoever implements it — "a subclass must implement this" stops holding once one has. The obligation to implement belongs in the class docstring, the not-yet-implemented fact in `raise NotImplementedError`, and the rationale in Internals.
- A lifecycle magic method (`__post_init__`, `__init_subclass__`) carries no docstring — a caller never invokes it directly, so the docstring would go unread. State the behavior it drives in the class docstring instead.
- A regular implementation module's docstring states the file's responsibility. When a matching Reference or Internals document exists, it must include a `See also:` entry pointing to that document.
- A package `__init__.py` docstring is optional. Keep one only for a user-facing import boundary, a CLI package whose purpose is not obvious, or a package-level behavior such as import-time registration. State only that package responsibility or public behavior in one to three sentences.
- An internal structural package or an `__init__.py` that only re-exports names has no docstring. Its path expresses the structure and `__all__` expresses the public names; a generic restatement of either adds no information.
- A package docstring carries no member catalog, design explanation, `See also:` entry, or other documentation link. Documentation discovery belongs to the documentation navigation rather than the import surface.
- In a regular implementation module, a pointer from code to a document appears only in the module docstring's `See also:` entry and is bare: the document path alone, with no section name, parenthetical, or prose. A function docstring and an inline comment carry no such pointer. A step comment aligned to a numbered procedure is not a document pointer and is exempt.
- Do not repeat a physical unit in prose when the name already carries a unit suffix. State a unit only when no suffix exists, the value is normalized or scaled, or the convention is otherwise non-obvious.

## Inline comments

- Use inline comments for local implementation help: non-obvious math, numerical intent, shape, and step markers.
- Design rationale, lifecycle, ownership, and cross-file contracts belong in Internals.

## Shape annotations

- Shape annotations are local implementation aids; the owning Reference or Internals document or the public docstring states the public shape contract.
- Do not hide a caller-visible shape contract in an inline comment.
- Add a shape annotation where a function or helper inserts, removes, merges, splits, permutes, reduces, broadcast-aligns, chunks, or reassembles axes. Skip pure elementwise code and code that operates at a known contract shape — `inst_shape`, a registered buffer shape, or an explicit per-call `shape` — unless it delegates to a helper that transforms layout. Annotate only where the shape changes or becomes non-obvious, not every tensor line.
- Use only these standalone forms, on the line above the tensor statement:

  ```python
  # Shape: [new]
  # Shape: [old] -> [new]
  ```

- An annotation contains only dimension names, integer literals, `*`, `->`, and `<name>=1` broadcast placeholders; a placeholder must name the semantic axis it aligns with. State only the shape; put any layout or ordering note in an ordinary comment on the line above.

## Step comments

Separate major procedural phases with this exact format, one blank line above and below:

```python

# --- 2: condense the cell network onto wire nodes ---

```

Use an integer or dotted hierarchical number, such as ``2`` or ``2.1``. Align each number and name with the ordered procedure in the module's Reference or Internals document.

## Type annotations

- Annotate every parameter and the return type in a function or method signature.
- Treat mypy as the baseline. When a false positive comes from an external library or a pattern mypy cannot express — a TypeVar not re-bound after an `isinstance` narrowing, a `fields()` or `replace()` call needing a `DataclassInstance` — leave the error unsuppressed; the project treats mypy as a helper. Never write `# type: ignore`.
- For base-class-related narrowing, fix the generic rather than reach for `cast`.
- Never add a meaningless runtime conversion only to satisfy typing.

## Config and policy dataclasses

- Declare config and policy descendants as ordinary classes inheriting `ConfigBase` or `PolicyBase`. The roots automatically apply `dataclass(frozen=True, kw_only=True)` to every descendant; never repeat `@dataclass` or write `__init__` on one.
- Declare every field with an annotation and document it under `Attributes:` in the class docstring. Follow the explicit-value and parameter-ownership rules in [config and policy](../internals/config_and_policy.md).
- Put config- and policy-domain checks in `validate()`. Both roots invoke the most-derived implementation after construction, so a descendant never declares `__post_init__` or calls `validate()` itself.
- Preserve inherited validation. A descendant extending a parent with constraints calls `super().validate()` or invokes the parent's named validation groups before its own checks.
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
