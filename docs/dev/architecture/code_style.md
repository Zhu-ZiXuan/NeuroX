# Code Documentation and Comment Style

This document defines the current NeuroX rules for code docstrings, inline comments, and shape annotations.

## Goals

Code carries three different kinds of written documentation:

- `docstring`: short, user-facing interface documentation.
- inline comments: local implementation help for reading the current code.
- `docs/dev/`: detailed developer-facing design documentation.

These three layers must stay separate.  The same design rule should not be repeated across all three places.

## Docstrings

### General rules

- Use Google-style docstrings.
- Keep docstrings local to the current symbol.
- Describe the current code and interface contract only.
- Do not store architectural history, design debates, or migration notes in docstrings.
- Do not describe how a higher-level module owns, builds, or uses the current module.

Detailed framework rules, design intent, coding conventions, and project-wide architecture belong under `docs/dev/`.

### Module docstrings

Module docstrings should only contain:

- the current responsibility of the file;
- an optional short `See also:` section that points to the corresponding `docs/dev/` file.

They should not contain long explanations of project structure or historical decisions.

### Class docstrings

Class docstrings should describe:

- what the class represents;
- the current external role of the class;
- for dataclasses, every field via `Attributes:`.

If a class owns child modules, it may describe those owned members only when that information is needed to understand the current class interface.

### Function and method docstrings

All functions and methods should document their interface with Google-style sections such as:

- `Args:`
- `Returns:`
- `Raises:` when needed

Rules:

- Every argument should be described.
- Every return value should be described.
- For physical quantities, include the unit and the meaning of the quantity.
- For non-physical quantities, describe the role of the value.
- If a tensor argument has a shape contract that matters at the current method level, include that shape in the argument description.

Docstrings should not explain higher-level workflows or global architecture.

#### Lifecycle prose

Method docstrings must not describe lifecycle policy, restart semantics, caching strategy, or state-holding strategy. Those belong in `fabrication_lifecycle.md` and `state_holding.md`. Each module's own lifecycle hooks (`_sample_fabricate_mismatch`, `program(...)`, `snapshot(...)`) should document their interface only — what they read / write, what shape — and leave cadence to the canonical doc.

```python
def _sample_fabricate_mismatch(self) -> None:
    """Resample β and V_th at ``self._inst_shape``."""
```

The full re-callability semantics live in `fabrication_lifecycle.md`; subclass override-points carry only the minimum interface signal.

## Inline comments

Inline comments should be used only for:

- local implementation details;
- shape annotations;
- step markers;
- mathematical or numerical details that help a reader understand the current block of code.

Inline comments should not be used for:

- architecture history;
- design debates;
- project-wide coding conventions;
- explanations of how higher-level modules use this code;
- long references to design documents.

## Shape annotations

Shape annotations are mandatory when tensor shapes change inside a function or method.

### Placement

- Shape annotations must appear only in code.
- They must be line comments directly above the corresponding tensor statement.
- They must not appear in `docs/dev/`.
- They must not appear in docstrings.
- `_sample_fabricate_mismatch` / `program` methods do not need shape annotations.

### Format

Use one of these two forms only:

- `# Shape: [new]`
- `# Shape: [old] -> [new]`

### Scope

- Include only the dimensions that matter at the current function or method level.
- Omit unrelated leading or trailing dimensions.
- Do not explain why the reshape exists inside the shape annotation itself.
- If a tensor shape does not change, no shape annotation is needed.
- If shapes do change in a function or method, every relevant tensor shape transformation should be annotated accurately.

### Allowed and disallowed content

Shape annotations may contain **only** dimension names, the `->` arrow, and broadcast-placeholder markers. Specifically:

- Allowed: dimension names, integer literals, the `*` product, and `<name>=1` placeholder dims for broadcast-only axes (kept tight: no spaces around `=`, no bold).
- Disallowed: ordering / layout commentary inside the shape comment (`(LSB first)`, `(data-major, digit-minor)`, …) — put such notes on a separate line above the shape comment if they are essential. Equality assignments other than the `=1` placeholder form. Side comments.

Broadcast-placeholder example (allowed):

```python
# Shape: [Bx, M, K] -> [Bx, M, K, Sa, digit_num=1]
```

Ordering / layout commentary (disallowed inside the shape comment):

```python
# LSB-first digit ordering.
# Shape: [...] -> [..., slice_num * digit_count]
```

## Step comments

When a comment is used to separate major procedural steps inside a function or method, use this exact format:

```python

# --- Step name and brief description ---

```

Rules:

- keep one blank line above the step comment;
- keep one blank line below the step comment;
- keep the step label short and local to the current code block;
- no trailing `#` after the closing `---`;
- numbered steps are allowed when the method contains a sequence of three or more ordered procedural phases (`# --- 1. Foo ---`, `# --- 2. Bar ---`, …); two-step or unordered blocks should use plain labels without numbers.

## Type annotations

Goal: every name carries a clear, accurate type for static checking — but only where the type is not already obvious from context. Redundant annotations duplicate information that the right-hand side already proves, and the duplication will eventually drift out of sync with reality.

### Always annotate

- **Function and method signatures**: every parameter and the return type.
- **Class-level attribute declarations** that are not paired with an immediate-and-obvious value at the declaration site: buffer / parameter type hints on `nn.Module` subclasses, dataclass fields, `Protocol` attribute declarations, forward declarations like ``self.solver: SolverType`` (no assignment).
- **Empty containers** at any scope: ``out: list[T] = []`` / ``out: dict[K, V] = {}`` / ``buf: tuple[int, ...] = ()``. Mypy infers these as ``list[Any]`` / ``dict[Any, Any]`` / ``tuple[()]`` without the annotation.

### Never annotate (drop the redundant one)

Inside function / method bodies, drop the annotation when the right-hand side trivially fixes the type:

- `self.foo = foo` when ``foo`` is a typed parameter or attribute.
- `self.count = len(xs)` — ``len`` returns ``int``.
- `self.x = 0` / `self.x = 0.0` / `self.x = ""` — literal of obvious type.
- `self.x = float(...)` / `int(...)` / `tuple(...)` — builtin constructor's return is unambiguous.
- `dcop = self.solver.solve_dc(...)` — the called method's typed return is the source of truth.
- `self.x = cfg.field * cfg.other` — arithmetic on already-typed scalars.

The same rule applies to local variables, not just `self.*` assignments.

### Why

Duplicating a type at the assignment site creates a second source of truth that maintenance will have to keep in lockstep with the signature, the dataclass field, or the call's return type. When the upstream type changes, the annotation at the assignment is the easiest one to forget — and mypy will keep accepting the stale annotation as long as it remains a supertype of the actual value. So redundant annotations don't make the code safer; they only add a quiet way to lie.

## FabricateMixin and buffer reassignment

Every module under `neurox/{device,analog,digital,xbar,macro}/` that owns fabricable state inherits [`FabricateMixin`](../modules/common/fabricate.md) alongside `nn.Module`. Subclasses override `_sample_fabricate_mismatch(self) -> None` only — never `fabricate()` itself. The mixin auto-cascades to every `FabricateMixin` child (including those wrapped in `nn.ModuleList` / `nn.ModuleDict`).

Buffer writes inside `_sample_fabricate_mismatch` and `program(...)` use **attribute reassignment**: `self.X = new_tensor`. Do **not** call `self.register_buffer("X", new_tensor, persistent=False)` a second time — reassignment updates the buffer slot in place via `nn.Module.__setattr__`, preserves the `persistent=False` flag, and keeps `.to(device)` / `.to(dtype)` migrations working. `register_buffer` is only used once per buffer at `__init__`.

## Per-instance shape parameter

The construction signature for fabricable modules ends with three runtime-context arguments after `cfg` / `name`:

```python
def __init__(self, *, cfg, name, <shape>, dtype, T__K) -> None: ...
```

The shape parameter name varies by layer:

- leaf circuits (analog / digital / device): `inst_shape: tuple[int, ...]` — per-instance fabrication shape.
- xbar tiles: `inst_shape: tuple[int, ...]` — per-instance multiplicity prefix. The xbar derives the trailing `(col_num, w_digit_count, row_num)` from its own cfg and exposes the full digit-tensor shape as `self._w_layout_shape`.
- xbar macros: `w_logical_shape: tuple[int, ...]` — operator-facing weight shape `(*prefix, N, K)`.

Every module stores `self._inst_shape: tuple[int, ...]` to satisfy `FabricateMixin`'s contract. At leaf and xbar level that is the constructor argument verbatim; at macro level it is conventionally `()`.

## Canonical docstrings for repeated members

A member that appears on many classes with identical signature **and identical semantic role** must use the same docstring text everywhere. Variation is reserved for cases where the member genuinely carries different information at that site.

### `__init__` parameters — one-line entries

In the `Args:` block of every fabricable-module constructor, the following parameters use the canonical line below. Do **not** rephrase per-class.

| Parameter | Canonical Args entry |
|---|---|
| `cfg` | `Concrete configuration dataclass.` |
| `name` | `Hierarchical instance name used by the profiler.` |
| `inst_shape` (leaf) | `Per-instance fabrication shape.` |
| `inst_shape` (xbar) | `Per-instance multiplicity prefix; trailing (col_num, w_digit_count, row_num) is derived from cfg.` |
| `dtype` | `Tensor dtype for internal buffers.` |
| `T__K` | `Operating temperature [K].` |
| `w_logical_shape` | `Logical weight shape (*prefix, N, K) bound to program(...).` |
| `ideal_xbar` | `When True, the macro replaces its physical xbar with the lossless ideal twin returned by xbar.to_ideal().` |

Per-subclass extra parameters keep their own descriptions; only the shared ones are fixed.

### Methods — full docstring

The following methods, when they appear on a class as a registry impl / override / abstract declaration, use exactly the docstring text below:

```python
# Family dispatcher (every Xbar / XbarMacro / ADC / DAC / ReadOut / TIA base).
@classmethod
def from_config(cls, *, cfg, ...) -> Self:
    """Build the concrete impl registered for ``type(cfg)``."""

# Leaf per-call snapshot (Driver, OpAmpTIA, NMOS, RRAM).
def snapshot(self, *, shape: tuple[int, ...]) -> <Name>Snapshot:
    """Sample one per-call runtime snapshot over ``shape``.

    Args:
        shape: Per-call broadcast shape; the snapshot fills tensor
            fields at this shape.

    Returns:
        Per-call snapshot of the fabricated state.
    """

# Macro static-weight write (Protocol, XbarMacro abstract, four XbarMacro impls — Direct / InterArraySlice / IntraArraySlice / Ideal).
def program(self, weight: Tensor) -> None:
    """Write the macro's static weight state from one logical weight tensor.

    Args:
        weight: Integer weight tensor whose shape matches
            ``self._w_logical_shape``.
    """

# Xbar static-weight write (Xbar abstract, IdealXbar, Offset1T1RXbar).
def program(self, w: Tensor) -> None:
    """Write the tile's owned device buffers from one xbar-native digit tensor.

    Args:
        w: Integer digit tensor whose shape matches
            ``self._w_layout_shape = (*inst_shape, col_num, w_digit_count, row_num)``.
            Entries must lie in :attr:`w_digit_range`.
    """

# Macro integer matmul (Protocol, XbarMacro abstract, four XbarMacro impls — Direct / InterArraySlice / IntraArraySlice / Ideal).
# Matches torch.matmul semantics (pure matmul, no bias). Bias add and
# requantize live in the operator layer.
def matmul(self, input: Tensor) -> Tensor:
    """Execute one integer matrix multiply against the programmed weight state.

    Matches ``torch.matmul`` semantics (pure matmul, no bias). Bias add
    and requantize live in the operator layer.

    Args:
        input: Integer activation tensor. Shape: ``[..., M, K]``.

    Returns:
        Integer pre-requantize output tensor. Shape: ``[..., M, N]``.
    """

# Xbar analog VMM (Xbar abstract, IdealXbar, Offset1T1RXbar).
def vec_mat_mul(self, x: Tensor) -> Tensor:
    """Run one analog VMM through the tile.

    Args:
        x: Activation tensor with primitive trailing ``[row_num]``.
            Entries must lie in :attr:`x_range`; leading dims are
            broadcast-only.

    Returns:
        ADC-code tensor with primitive trailing ``[col_num]``.
    """
```

### Properties — one-line entries

```python
# PPA on every fabricable leaf (analog / digital / device / xbar tile).
@property
def area_per_inst__um2(self) -> float:
    """Silicon area per instance [um^2]."""

@property
def leakage_per_inst__uW(self) -> float:
    """Static leakage per instance [uW]."""

@property
def latency_per_op__ns(self) -> float:
    """Latency per op [ns]."""

# Macro value-domain (Protocol, XbarMacro abstract, four XbarMacro impls — Direct / InterArraySlice / IntraArraySlice / Ideal).
@property
def w_value_range(self) -> tuple[int, int]:
    """Inclusive integer weight range accepted by the macro."""

@property
def x_value_range(self) -> tuple[int, int]:
    """Inclusive integer activation range accepted by the macro."""

@property
def output_rescale_factor(self) -> float:
    """Ratio of the ideal partial-product max to the actual tile output max."""
```

**Exception — `latency_per_op__ns(*, bits/adc_bits)`**: ADC and Readout publish a parametric latency (per-conversion or per-VMM-pipeline). Keep the site-specific extended docstring because the signature carries a `bits` keyword the canonical short form cannot describe.

**Exception — `w_digit_range` on `Offset1T1RXbar`**: the offset-coded array's docstring explains the `(-o, S - 1 - o)` derivation, which is offset-specific and worth keeping verbatim.

### What this does not mandate

- Implementation-specific notes (e.g. "Aggregated leakage rolls up from children", "Settling-dominated") may live in the class docstring or in inline comments, but **must not** displace the canonical property docstring text.
- An override that simply delegates (`return self.cfg.area_per_inst__um2`) may either repeat the canonical one-liner or omit the docstring entirely and inherit from the base / abstract — pick one rule per family and apply it consistently.

## Property vs method

The public surface of a class distinguishes properties from methods by the **nature of the returned value**, not by ergonomic preference.

### Rules

- Use `@property` if **all three** hold:
  - the return is fully determined at `__init__` time (or by intrinsic instance state),
  - it has no dynamic component (no per-call input dependency, no time-varying ambient state),
  - accessing it does not mutate the instance.
- Use a method in every other case. Concretely, return from a method when the value depends on the call's arguments, when it depends on time-varying ambient state, or when computing it mutates the instance.

### Where a value lives

- A value that is fixed at construction time, or that simply names a piece of intrinsic instance state, lives as a `@property` on the instance.
- A value that is dynamically determined by a method's inputs is returned from that method.

### Caveat — no synthetic state

A dynamic, one-shot value must not be cached into instance state purely so that it can be exposed as a property. That dishonestly promotes a method-shaped operation into the property surface and grows the instance's apparent state. If a value cannot be derived without per-call input, keep it as a method return.

## Physical-layer no defaults

Inside the physical layer — every module under `neurox/device/`, `neurox/analog/`, `neurox/digital/`, `neurox/xbar/`, and `neurox/macro/` — the surface is strictly no-default:

- **Config dataclass fields** carry no default. Every chip TOML must declare every field explicitly. `field(default_factory=...)` is also forbidden.
- **Function and method parameters** (including `__init__`) carry no default. Every call site spells out every argument. Applies to every callable, not just constructors.

The rule applies to the source-of-truth declarations under the physical layer. Class-level `Final`-style constants (e.g. `N_NEWTON: int = 4` used as `self.N_NEWTON`) are not parameters and not config fields — they stay as written. Local-variable type annotations on empty containers (`out: list[int] = []`) are also untouched; see *Always annotate* above.

### Why

A default value is a hidden second source of truth. When a TOML omits a PPA field, the implicit `0.0` quietly disables the contribution; when a call site omits a kwarg, the implicit value silently picks a behaviour. Both break under refactor and both hide configuration drift. Forcing every value into the call site makes the active configuration legible and makes "what did this run actually use" answerable from the inputs alone. The cost is verbosity at construction; the buyback is no more silent regressions through stale defaults.

### When you need to add a field

Adding a field to a physical-layer config or callable means updating every chip TOML and every call site in the same change. There is no "add with a default, migrate later" path. If a field is truly optional, model it explicitly with `Optional[T]` plus an enable toggle (the noise / mismatch pattern below).

## Noise / mismatch configuration

Non-ideality, mismatch, and dynamic-noise fields follow a separate uniform rule documented in [`noise_and_toggles.md`](noise_and_toggles.md): every source carries a fully-populated parameter and a paired `enable_<source>: bool` toggle; `None` is forbidden in cfg fields; runtime helpers in `neurox/common/nonideality.py` take an `*, enabled: bool` kwarg.

## Relationship with `docs/dev/`

`docs/dev/` is the canonical home for:

- architecture rules;
- design intent;
- coding conventions;
- ownership and construction rules;
- module interaction rules;
- algorithmic rationale;
- historical decisions and ADRs.

Code should only contain enough written context to read and use the current implementation.

## Documentation dependency direction

Documentation must follow the same dependency direction as the code: an **upper module may reference lower modules**, but a **lower module must not reference any upper module**. This applies uniformly to module docstrings, inline comments, and `docs/dev/modules/<file>.md`.

A "lower module" is one that the other side imports / constructs / owns; an "upper module" is the consumer side of that relationship. The dependency graph in the code defines which side is which.

### Why this rule exists

- A lower module is typically reused by many upper modules. Naming any one consumer in a lower module's docs creates a maintenance burden every time a new consumer appears.
- Lower modules are supposed to be the stable layer. Mixing upper-module names into their docs raises their effective change rate to match the upper modules.
- Readers of a lower module want to know what it does and how to use it. Naming a specific consumer narrows their mental model of the module's generality.
- Following the rule makes the doc dependency graph acyclic — the same property the code dependency graph already has.

### What lower-module docs (and code) may say

- The module's own current responsibility.
- Its current input / output contract, units, invariants.
- The protocol surface it exposes, described abstractly (e.g. "given `(V_g, V_d, V_s)` return `I_ds` and partials" — without naming a specific solver).
- Cross-references to other same-layer or lower-layer artefacts (other files in the same family, `architecture/` rules, ADRs).
- Generic terms for callers: "the consuming circuit", "the calling solver", "any caller that respects the protocol", etc.

### What lower-module docs must not say

- The name of any specific upper module.
- Sentences of the form "is used by X", "is owned by X", "is the default in the Y path".
- Workflow descriptions that step through an upper module's lifecycle.

### Where ownership relationships live instead

Each upper-module doc carries its **own** description of:

- which child modules / configs it owns;
- the role each child plays inside it;
- how it constructs each child (direct `__init__` vs. family `from_config`);
- how it propagates the runtime trio (`name`, `T__K`, `dtype`) and any family-specific extras (`ideal_xbar`, …) into its children.

The "who uses who" graph is therefore asserted exactly once, from the parent side.

### Edge cases

- **Abstract base / Protocol docs** are not violations when they describe the contract any future consumer must respect — they intentionally name no specific consumer.
- **Intra-family references** are allowed: `docs/dev/modules/analog/dac/general.md` may say it is a concrete `DAC` family impl registered via `DAC.from_config`. That is same-layer reference, not upward.
- **`architecture/` and `adr/`** intentionally take a global view and may cross layers freely. The rule applies to `modules/` and to in-code docstrings / comments only.
- **Examples in lower-module docs** should use generic placeholder names (`SomeCircuit`, `parent`) rather than naming a specific upper module.

### Directory-level READMEs

`docs/dev/modules/<path>/README.md` files (the directory-level family / package overviews) **are allowed** to:

- enumerate the files / classes that live in the directory;
- describe the same-layer family layout (e.g. "this directory holds the `*Slicer` family: `SerialSlicer`, `SimpleSlicer`");
- describe ownership of sub-directories (e.g. "the `_1t1r/` subdirectory holds the 1T1R cell topology").

They must **not** drift into per-file workflow narration or per-class upper-consumer descriptions — those belong on the corresponding leaf file's dev doc (and must follow the dependency-direction rule there).

When a new class is added to or removed from a directory whose README enumerates its members, the README **must** be updated in the same change. Stale enumerations are treated as a documentation regression.
