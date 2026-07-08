# Writing mixin documents

## Scope

A mixin's documentation is split across two carriers. The class and method docstrings carry the contract — what a host must provide, what a subclass overrides, what fires automatically — because that is what a reader reads at the code. The `.md` document carries what no docstring should: the design rationale, the rejected alternatives, and the composition. A mixin lives under `internals/common/mixin/` and names no concrete host.

## Where each thing goes

The line is whether a reader can read it off the code itself:

- **Method docstring** — a called API, an overridable hook and its default-body semantics, and a single-method contract (`Raises`, `Returns`). A reader who calls or overrides a method reads its docstring, so that is the home.
- **Magic method** (`__init_subclass__`, `__post_init__`) — no docstring; a reader never visits one, so its behavior goes to the class docstring instead.
- **Class docstring** — only the two contracts no single method carries: what the host must do, and what fires automatically.
- **`.md`** — the design rationale, the rejected alternatives, the cross-call invariants told as *why*, and the composition.

## Class docstring

Open with the capability the mixin grants a host and its boundary, in role language, then carry the two contract sections:

```python
"""<Capability granted to a host, and its boundary — role language>.

Host requirements:
    - <a base to inherit, an attribute to set, a method or factory to
      declare, a type parameter to bind, or a call convention to honor>.

Injected behavior:
    - <what fires with no explicit call — an ``__init_subclass__`` or
      ``__post_init__`` effect, auto-registration — named with its trigger>.
"""
```

- `Host requirements` almost always has content — a mixin exists to be composed, so it demands something of its host.
- `Injected behavior` appears only when the mixin hooks a lifecycle magic method; a pure-helper or explicit-call mixin has none, so omit the section — never write `None`.
- A method the host must write itself — an abstract hook, or one the mixin requires it to declare — goes in `Host requirements` as a bare "implement `X`", never with its semantics restated (those are in the method's docstring). A method the host only calls is left to its signature and docstring. So there is no `Override surface` or `Provided API` section: the only question is whether the host writes the method or calls it.

## `.md` template

```markdown
# <Mixin> mixin

## Design decisions

## Composition

---

- **Reference**: N/A — software mechanism
- **Implementation**: `neurox/common/mixin/<...>.py`
- **Tests**: `tests/test_<...>.py` or TODO — <what is missing>
```

- Lead paragraph [optional]: the capability and boundary in role language — the `.md`'s own entry point, not a verbatim copy of the class docstring; keep it only when it synthesizes more than the H1, and do not add a `## Summary` heading.
- `Design decisions` [usually present]: the software rationale — what the host owns versus the mixin, any rejected alternative, and the cross-call invariants (re-callable without accumulation, emit-once, per-family isolation) told as *why it is so*, not restated as a contract.
- `Composition` [on demand]: the mixin's position in the host MRO, its ordering relative to other mixins, and its trigger timing.
- The footer is traceability; `Reference` is always `N/A — software mechanism`.

## Content rules

### No downward links

A host composes a mixin, so the mixin is the lower module and its hosts are the upper consumers. State every requirement as a role the host fills — the base it must inherit, the attribute it must set — never as a named host module. Pointing a mixin at a specific host reverses the dependency direction.

**Bad:** "`SchemeFamily` sets `_inst_shape` before the cascade runs."

**Good:** "The host sets `_inst_shape` in its `__init__` before the cascade runs."

### Single contract home

Each contract has one home: a method's in its own docstring, a host-side or injected contract in the class docstring, the design in the `.md`. Nothing restates another's home — the `.md` cites a contract rather than re-listing it, and a host or subclass document states only how it composes or specializes the mixin.
