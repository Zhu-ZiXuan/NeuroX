# Validate mixin

## Summary

`ValidateMixin` grants a host a small set of runtime-check helpers — strict-positivity, non-negativity, strict monotonicity, and minimum-length — that the host calls through `self` from inside its own `validate*` methods to assert the runtime constraints static typing cannot express. It is a pure helper: it owns no field, no construction hook, and no `validate` of its own, so inheriting it supplies the checks but does not by itself make a host validated — running validation at construction is the host's to wire. It carries no physics.

## Design decisions

- **The mixin is pure — it ships only check helpers, never `__post_init__` or `validate`.** Keeping both the *when* (construction-time validation) and the *which* (the set of checks a host runs) out of the mixin puts those in each host rather than buried in shared mixin source; the mixin answers only "how do I assert this one constraint".
- **Helpers are `@staticmethod`, called through `self`.** They hold no instance state and read nothing from the host — invoking through `self` is only for call-site readability. Each helper is a self-contained predicate-plus-message, so a host can combine several into one `validate_<group>` without coupling.
- **Checks encode only what static analysis cannot.** Numeric bounds, monotonicity, length, and cross-field relationships are the mixin's domain; type membership, `Literal` sets, and tensor shape or dtype are the annotation's job. The helper set stays deliberately narrow and grows only when a genuinely new class of runtime constraint appears.
- **Failures raise immediately with a uniform message shape.** Each helper raises `ValueError` carrying the offending name and value rather than collecting a report; the first violated constraint stops construction, so a host can never come into existence in a partially-valid state.

## Composition

Because the mixin carries no construction or class-creation hook, it constrains no method-resolution order, and a host may compose it in any position.

---

- **Reference**: N/A — software mechanism
- **Implementation**: `neurox/common/mixin/validate.py`
- **Tests**: TODO — no dedicated `ValidateMixin` test; `validate*` failures are exercised indirectly through subsystem config tests such as `tests/test_adc_family.py` and `tests/test_signal_chain.py`.
