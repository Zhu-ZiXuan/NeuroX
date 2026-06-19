# Code Style

This document defines the rules for code docstrings, inline comments, and shape annotations. Document-prose format (characters, markdown, math rendering) is in [doc_style](doc_style.md); code-side unit suffixes (`__ns`, `__K`) and dtype policy are covered here, while the equation-side symbols and units for the same quantities are in [reference/notation_conventions](../reference/notation_conventions.md).

## Goals

Code carries three kinds of written documentation:

- `docstring`: short, user-facing interface documentation.
- inline comments: local implementation help for reading the current code.
- developer docs: detailed design documentation in the reference and internals trees.

These three layers stay separate. The same design rule is not repeated across all three.

## Docstrings

### General rules

- Use Google-style docstrings.
- Keep docstrings local to the current symbol.
- Describe the current code and interface contract only.
- Do not store architectural history, design debates, or migration notes in docstrings.
- Do not describe how a higher-level module owns, builds, or uses the current module.

Framework rules, design intent, and project-wide architecture belong in the developer docs, not in docstrings.

### Module docstrings

A module docstring contains only the current responsibility of the file, plus an optional short `See also:` pointing to the corresponding reference or internals document. It does not explain project structure or historical decisions.

### Class docstrings

A class docstring describes what the class represents, its current external role, and — for dataclasses — every field via `Attributes:`. It may describe owned child modules only when that is needed to understand the current class interface.

### Function and method docstrings

Document the interface with Google-style sections (`Args:`, `Returns:`, `Raises:` when needed). Every argument and return value is described; for a physical quantity give the unit and meaning, for a non-physical value give its role; if a tensor argument has a shape contract that matters at this method level, state the shape. Do not explain higher-level workflows or global architecture.

#### Lifecycle prose

A method docstring does not describe lifecycle policy, restart semantics, caching, or state-holding strategy — those are fabrication-lifecycle and state-holding conventions kept in the developer docs. A module's own lifecycle hooks (`_sample_fabricate_mismatch`, `program(...)`, `snapshot(...)`) document their interface only — what they read / write, at what shape:

```python
def _sample_fabricate_mismatch(self) -> None:
    """Resample mismatch at ``self._inst_shape``."""
```

Cadence and re-callability are left to the lifecycle convention; an override point carries only the minimum interface signal.

## Inline comments

Use inline comments only for local implementation details, shape annotations, step markers, and mathematical / numerical notes that help read the current block. Do not use them for architecture history, design debates, project-wide conventions, descriptions of how higher-level modules use this code, or long references to design documents.

## Shape annotations

Shape annotations are mandatory when tensor shapes change inside a function or method.

### Placement

- Only in code, as line comments directly above the corresponding tensor statement.
- Never in docstrings or developer docs.
- `_sample_fabricate_mismatch` / `program` methods do not need them.

### Format

Use one of these two forms only:

- `# Shape: [new]`
- `# Shape: [old] -> [new]`

### Scope

- Include only the dimensions that matter at the current level; omit unrelated leading/trailing dims.
- Do not explain why the reshape exists inside the shape annotation.
- Annotate every relevant transformation when shapes change; none when they do not.

### Allowed and disallowed content

A shape annotation contains **only** dimension names, integer literals, the `*` product, the `->` arrow, and `<name>=1` broadcast-placeholder dims (no spaces around `=`).

- Disallowed: ordering / layout commentary inside the shape comment (`(LSB first)`, `(data-major, digit-minor)`); put such a note on a separate line above. Equality assignments other than the `=1` placeholder. Side comments.

```python
# LSB-first digit ordering.
# Shape: [Bx, M, K] -> [Bx, M, K, Sa, digit_count=1]
```

## Step comments

To separate major procedural steps inside a function, use this exact format:

```python

# --- Step name and brief description ---

```

- One blank line above and below; short, local label; no trailing `#`.
- Numbered steps (`# --- 1. Foo ---`) only when there are three or more ordered phases; otherwise plain labels.

## Type annotations

Every name carries a clear type for static checking — but only where it is not already obvious from context. A redundant annotation duplicates what the right-hand side already proves and eventually drifts out of sync.

### Always annotate

- Function / method signatures: every parameter and the return type.
- Class-level attribute declarations not paired with an immediate obvious value (buffer / parameter hints on `nn.Module`, dataclass fields, `Protocol` attributes, forward declarations like `self.solver: SolverType`).
- Empty containers at any scope: `out: list[T] = []`, `buf: tuple[int, ...] = ()` (mypy infers `Any` otherwise).

### Never annotate (drop the redundant one)

Inside bodies, drop the annotation when the right-hand side trivially fixes the type: `self.foo = foo` (typed param), `self.count = len(xs)`, `self.x = 0` / `0.0` / `""`, `self.x = float(...)`, `dcop = self.solver.solve_dc(...)`, arithmetic on typed scalars. Applies to locals as well as `self.*`.

### Why

Duplicating a type at the assignment site creates a second source of truth that maintenance must keep in lockstep; when the upstream type changes, the assignment annotation is the easiest to forget, and mypy keeps accepting a stale supertype. Redundant annotations do not make code safer — they add a quiet way to lie.

## Dtype on the runtime path

Each computation domain has a fixed dtype. Dtype must not change between calls of the same `@torch.compile`-decorated function — a dtype change forces a recompile. The equation-side units for these same paths are in [reference/notation_conventions](../reference/notation_conventions.md).

| Domain | Dtype | Notes |
|---|---|---|
| Digital computation | `int32` | Standard MAC / shift / requantize result. |
| Digital table indexing | `int64` | Only for table lookups. |
| Analog forward path | chosen at construction, calibrated per build | `bfloat16` may be insufficient when low-LSB noise must stay visible; the calibration tool fixes the analog dtype per build. |
| Solver internals | hard-coded per solver, calibrated | Determined by the calibration tool; the solver does not switch dtype at runtime. |
| Output rescale (multiplier / rshift) | `float32` | Data fits in `int16`; `int32` is used only for GPU friendliness, not range. |

## FabricateMixin and buffer reassignment

Every module under `neurox/{device,analog,digital,xbar,macro}/` that owns fabricable state inherits `FabricateMixin` (`neurox/common/fabricate.py`) alongside `nn.Module`. Subclasses override `_sample_fabricate_mismatch(self) -> None` only — never `fabricate()`. The mixin auto-cascades to every `FabricateMixin` child.

Buffer writes inside `_sample_fabricate_mismatch` and `program(...)` use **attribute reassignment** (`self.X = new_tensor`), not a second `register_buffer(...)` call: reassignment updates the buffer slot via `nn.Module.__setattr__`, preserves `persistent=False`, and keeps `.to(...)` migrations working. `register_buffer` is used once per buffer, at `__init__`.

## Per-instance shape parameter

The construction signature for fabricable modules ends with three runtime-context arguments after `config` / `name`:

```python
def __init__(self, *, config, name, <shape>, dtype, T__K) -> None: ...
```

The shape parameter name varies by layer: leaf circuits and xbar tiles take `inst_shape: tuple[int, ...]`; xbar macros take `w_logical_shape: tuple[int, ...]`. Every module stores `self._inst_shape` to satisfy `FabricateMixin` (the constructor argument verbatim at leaf/xbar level; conventionally `()` at macro level). No parameter carries a default (see *Physical-layer no defaults*).

## Canonical docstrings for repeated members

A member that appears on many classes with identical signature **and identical semantic role** uses one docstring text everywhere; variation is reserved for sites that genuinely carry different information. The canonical text is written once — on the abstract base, or as the shared `Args:` line — and concrete impls do not rephrase it.

- **Shared `__init__` parameters** (`config`, `name`, `inst_shape`, `dtype`, `T__K`, ...) use one fixed `Args:` line each. Example:

```python
config: Concrete configuration dataclass.
```

- **Repeated family methods** (`from_config`, `snapshot`, `program`, `matmul`, `vec_mat_mul`, `adc_rescale_factor`, ...) carry the docstring written on their abstract declaration; an impl restates it only when its behaviour genuinely differs (e.g. `Offset1T1RXbar.w_digit_range`, whose offset-specific derivation earns its own text).
- **Inherited PPA properties** (`area_per_inst__um2` / `leakage_per_inst__uW`) are defined once on `CircuitBase`; leaf circuits do not redeclare them.

Implementation-specific notes may live in the class docstring or inline comments, but must not displace the canonical text.

This document deliberately does not reproduce the canonical wording: the text lives once on the abstract base classes in `neurox/`, and contributors consult those base-class docstrings for the exact phrasing rather than a catalog kept here.

## Property vs method

A class distinguishes a property from a method by the **nature of the returned value**, not by ergonomic preference.

- Use `@property` only if all three hold: the return is fully determined at `__init__` (or by intrinsic instance state), it has no per-call or time-varying dependency, and accessing it does not mutate the instance.
- Use a method otherwise — when the value depends on the call's arguments, on time-varying ambient state, or when computing it mutates the instance.

**No synthetic state**: do not cache a dynamic one-shot value into the instance just to expose it as a property. That promotes a method-shaped operation into the property surface and grows the apparent state. If a value needs per-call input, keep it a method.

## Physical-layer no defaults

Inside the physical layer — every module under `neurox/{device,analog,digital,xbar,macro}/` — the surface is strictly no-default:

- **Config dataclass fields** carry no default (and no `field(default_factory=...)`); every chip TOML declares every field.
- **Function and method parameters** (including `__init__`) carry no default; every call site spells out every argument.

Class-level `Final`-style constants and empty-container local annotations are not parameters and stay as written.

### Why

A default is a hidden second source of truth: an omitted PPA field silently becomes `0.0` and disables a contribution; an omitted kwarg silently picks a behaviour. Forcing every value to the call site makes the active configuration legible and "what did this run use" answerable from the inputs alone. There is no "add with a default, migrate later" path — adding a field updates every TOML and call site in the same change. A truly optional field is modelled explicitly with `Optional[T]` plus an enable toggle (the noise / mismatch pattern).

## Noise / mismatch configuration

Non-ideality, mismatch, and dynamic-noise fields follow a separate uniform rule: every source carries a fully-populated config parameter and a paired `bool` field on the module's `*Policy` dataclass; `None` is forbidden in config fields; the runtime helpers (`neurox/common/nonideality.py`) take an `*, enabled: bool` kwarg.

## Documentation dependency direction

Documentation follows the same dependency direction as the code: an **upper module may reference lower modules**, but a **lower module must not reference any upper module**. This applies to module docstrings, inline comments, and the per-module reference / internals documents. (Convention and term documents are the sink: every specific document references them, and they reference nothing more specific — see [doc_style](doc_style.md).)

A "lower module" is one the other side imports / constructs / owns; an "upper module" is the consumer. The code dependency graph defines which is which.

### Why

A lower module is reused by many upper modules; naming any one consumer creates churn when a new consumer appears, raises the lower module's effective change rate, and narrows a reader's sense of its generality. The rule keeps the doc dependency graph acyclic, like the code's.

### What lower-module docs may say

- Its own responsibility, I/O contract, units, invariants.
- Its protocol surface, described abstractly (e.g. "given `(V_g, V_d, V_s)` return `I_ds` and partials" — no specific solver named).
- Cross-references to same-layer or lower-layer artefacts.
- Generic terms for callers: "the consuming circuit", "any caller that respects the protocol".

### What lower-module docs must not say

- The name of any specific upper module; sentences like "is used by X" / "is owned by X" / "is the default in the Y path"; a walk-through of an upper module's lifecycle.

### Where ownership lives instead

Each upper-module doc carries its own description of which children it owns, the role of each, how it constructs each (direct `__init__` vs family `from_config`), and how it propagates the runtime trio. The "who uses who" graph is asserted once, from the parent side.

### Edge cases

- Abstract base / Protocol docs that describe the contract any future consumer must respect are not violations.
- Intra-family references are allowed (a concrete family impl may say it registers via `Family.from_config`) — same-layer, not upward.
- Examples in lower-module docs use generic placeholder names (`SomeCircuit`, `parent`), never a specific upper module.

### Directory-level READMEs

A directory `README.md` (the family / package overview) may enumerate the files / classes in the directory, describe the same-layer family layout, and describe ownership of its sub-directories. It must not drift into per-file workflow narration or per-class upper-consumer descriptions. When a class is added to or removed from a directory whose README enumerates members, update the README in the same change.
