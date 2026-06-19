# ValidateMixin — Implementation

## Summary

`ValidateMixin` is a pure helper mixin for frozen `*Config` dataclasses. It contributes a small set of static check helpers — strict-positivity, non-negativity, strict monotonicity, and minimum-length — that a config calls through `self` from inside its own `validate*` methods. The mixin carries no construction hook and no `validate` of its own: the project-wide rule that validation runs at construction (`__post_init__` calling `self.validate()`) lives in each config and family base, not in the mixin. This is software plumbing with no physics-bearing reference counterpart.

## Design decisions

- **The mixin is pure — it ships only check helpers, never `__post_init__` or `validate`.** Keeping the *when* (construction-time validation) out of the mixin means the entry-point rule is documented and enforced in each family base, not buried in shared mixin source. The mixin answers only "how do I assert this single constraint", leaving "when do checks run" and "which checks exist" to the config tree.

- **Helpers are `@staticmethod`, called through `self`.** A config invokes a helper through `self` for readability, but the helpers hold no instance state and depend on nothing from the config. This keeps each helper a self-contained predicate-plus-message and lets a config mix several into one `validate_<group>` without coupling.

- **Checks encode only what static analysis and the loader cannot.** Numeric bounds, monotonicity, length, and cross-field relationships are the mixin's domain; type membership, `Literal` sets, and tensor shape / dtype are not — those are the annotation's job and the loader's job. The helper set is intentionally narrow for this reason and grows only when a genuinely new class of runtime constraint appears.

- **Failures raise immediately with a uniform message shape.** Each helper raises `ValueError` carrying the offending field name and value, rather than collecting failures into a report. The first violated constraint stops construction, so a config can never come into existence in a partially-valid state.

## Contracts & invariants

- **Every frozen config inherits `ValidateMixin`** (directly or transitively via `CircuitConfig`) and reaches the helpers through `self`. The mixin contributes no fields and no `__post_init__`; a config that inherits it still owns its own `__post_init__` (calling `self.validate()` and nothing else) and its own `validate`.

- **Helpers raise `ValueError` on violation and return `None` on success.** Each takes the value(s) under test plus a human-readable `name` used in the error message. A passing check is silent; there is no success return value to inspect.

- **Provided checks (developer-facing contract):**
  - require a scalar be strictly positive (`value > 0`);
  - require a scalar be non-negative (`value >= 0`);
  - require a sequence be strictly increasing;
  - require a sequence be strictly decreasing;
  - require a sequence meet a minimum length.

  Sequence checks accept any iterable of `int | float`; the monotonicity checks compare consecutive elements pairwise, so an empty or single-element sequence trivially passes.

- **A config groups its checks into `validate_<group>` methods, orchestrated by its own `validate`.** Multi-field configs split into one `validate_<group>()` per logical field group (the standard name `validate_ppa` covers the area / leakage / latency trio); single-field configs may inline the check. A subclass `validate` opens with `super().validate()` so validation chains up the inheritance tree. None of this orchestration is contributed by the mixin — it lives in the config and family base. See [config_and_construction](../../config_and_construction.md).

- **Do not re-validate inner distribution sub-configs.** A config holding a probability-distribution sub-config must not re-run the helpers against the sub-config's fields in its own `validate*`; the inner config's `__post_init__` already validated on construction.

## Performance & resources

N/A — validation runs once per config instance at construction, off any per-VMM hot path. Sequence checks are a single linear pass over the iterable.

## Gotchas

- **The mixin does not make a config validated.** Inheriting `ValidateMixin` only supplies helpers; a config that forgets its own `__post_init__ → validate()` wiring inherits no construction-time check. The mixin cannot enforce the entry point because it deliberately carries no construction hook.

- **The monotonicity helpers pass trivially on short sequences.** An empty or single-element sequence satisfies both strictly-increasing and strictly-decreasing checks, because there is no adjacent pair to violate. Pair a length check with a monotonicity check when a non-degenerate sequence is required.

- **Helper bounds are strict or inclusive as named — do not assume.** Positivity is strict (`> 0`), non-negativity is inclusive (`>= 0`), and both monotonicity helpers are strict (equal neighbours fail). Pick the helper whose boundary matches the physical constraint rather than relying on a default.

## Known limitations

- **No structural guarantee that a config wires validation at all.** Whether `__post_init__` calls `self.validate()` is a per-config convention, checked at construction and by tests, not by the mixin. A config can inherit the mixin and still skip validation entirely.

- **Failures are reported one at a time, not aggregated.** Construction stops at the first violated constraint; a config with several independent violations surfaces them across successive fix-and-rerun cycles rather than in a single report.

---

- **Reference**: N/A — software plumbing; no physics counterpart.
- **Implementation**: `neurox/common/mixin/validate.py`
- **Tests**: `tests/test_config_validation.py`
- **Decisions**: [config_and_construction](../../config_and_construction.md)
