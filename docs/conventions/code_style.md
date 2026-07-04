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

[notation_conventions](notation_conventions.md) is the authority for the non-ASCII whitelist and for where a formula may live; this section states only how that policy applies inside a code file.

**Effective code is ASCII.** Identifiers and protocol or data string literals use only ASCII. Python 3 accepts unicode identifiers, so this is an enforced rule, not an automatic property.

**A docstring or comment may carry raw whitelisted unicode** — a Greek variable (σ, μ, τ), a partial derivative (∂I/∂V), a superscript power (x²) — as the glyph itself. Two limits apply:

- **No hosted formula.** A docstring or comment holds at most simple inline notation: a lone symbol, a short inline expression, an inline ∂I/∂V. A multi-term derivation lives in the module's Reference or Internals document, and the docstring points there.
- **No LaTeX.** A comment never renders and a docstring is read as plain text first, so write the symbol directly (σ), never its LaTeX form (`$\sigma$`); LaTeX is a Markdown-only tool.

## Docstrings

- Use Google-style docstrings.
- Write for the caller of the symbol: public semantics, and tensor shapes when shape is part of the public contract. A pure elementwise API may omit shape or state the same-shape rule once.
- Write the complete interface docstring on the abstract base, mixin, or Protocol. An unchanged override inherits it instead of copying it; document only the difference when an override changes contract, shape, side effects, units, or errors.
- An interface docstring states what the method does, not a directive to whoever implements it — "a subclass must implement this" stops holding once one has. The obligation to implement belongs in the class docstring, the not-yet-implemented fact in `raise NotImplementedError`, and the rationale in Internals.
- A lifecycle magic method (`__post_init__`, `__init_subclass__`) carries no docstring — a caller never invokes it directly, so the docstring would go unread. State the behavior it drives in the class docstring instead.
- A module docstring states the file's responsibility. When a matching Reference or Internals document exists, it must include a `See also:` entry pointing to that document.
- A pointer from code to a document appears only in that `See also:` entry, and is bare: the document path alone, with no section name, parenthetical, or prose. A function docstring and an inline comment carry no such pointer; a reader reaches the spec through the file's single `See also:`. A step comment aligned to a numbered procedure is not a document pointer and is exempt.
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

# --- Step 2: condense the cell network onto wire nodes ---

```

Number the steps and align each number and name with the ordered procedure in the module's Reference or Internals document. When the code directly implements a numbered procedure, the step numbers and names match that document.

## Type annotations

- Annotate every parameter and the return type in a function or method signature.
- Treat mypy as the baseline. Manually ignore a false positive caused by an external library or a pattern mypy cannot express; do not write `# type: ignore`.
- For base-class-related narrowing, fix the generic rather than reach for `cast`.
- Never add a meaningless runtime conversion only to satisfy typing.

## Property vs method

Use `@property` only for a value fixed by construction, computed with at most a cheap, side-effect-free expression over init-fixed inputs (e.g. a shape product). The allowed cases are:

- Config-field accessor: exposes a field of the owned config.
- Transparent delegation: forwards to an attribute of an owned object.
- Abstract base or Protocol contract: declares an attribute-shaped interface point.
- One-step arithmetic over init-fixed state: a single cheap expression.

Anything that touches a runtime tensor, performs real computation, or has a side effect is a method, as is anything that depends on call arguments, runtime mode, or mutation.

## Control flow

Do not branch on tensor values. A data-dependent branch forces a graph break under the caller's `torch.compile`; express the choice with tensor operations such as masking, `torch.where`, or indexing. When a value-dependent branch is unavoidable, explain the reason in an inline comment.
