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

Method docstrings must not describe lifecycle policy, restart semantics, caching strategy, or state-holding strategy. Those belong in `fabrication_lifecycle.md` and `state_holding.md`. The single exception is the `(re-callable)` keyword tag at the end of a `fabricate(...)` summary line, which is allowed and recommended as the minimum interface signal:

```python
def fabricate(self, shape: tuple[int, ...]) -> None:
    """Sample static per-instance state over ``shape`` (re-callable).

    Args:
        shape: Per-instance fabrication shape.
    """
```

The full re-callability semantics live in `fabrication_lifecycle.md`; the keyword in the docstring is a pointer, not a restatement.

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
- `fabricate` methods do not need shape annotations.

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

- The name of any specific upper module (`CircuitCore1T1R`, `XbarMacro`, `OffsetSwitchCapMuxAdcReadOut`, …).
- Sentences of the form "is used by X", "is owned by X", "is the default in the Y path".
- Workflow descriptions that step through an upper module's lifecycle.

### Where ownership relationships live instead

Each upper-module doc carries its **own** description of:

- which child modules / configs it owns;
- the role each child plays inside it;
- how it constructs each child (direct `__init__` vs. family `from_config`);
- how it propagates the runtime trio (`name`, `T__K`, `dtype`) and any family-specific extras (`stochastic`, …) into its children.

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
