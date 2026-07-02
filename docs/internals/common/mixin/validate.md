# Validate mixin

## Summary

`ValidateMixin` is a pure helper mixin for frozen `*Config` dataclasses. It contributes a small set of static check helpers — strict-positivity, non-negativity, strict monotonicity, and minimum-length — that a config calls through `self` from inside its own `validate*` methods. The mixin carries no construction hook and no `validate` of its own; running validation at construction — a `__post_init__` calling `self.validate()`, which dispatches grouped `validate_<group>()` checks chained by `super().validate()` — is the orchestration convention every config and family base follows. This is software plumbing with no physics-bearing reference counterpart.

## Design decisions

- **The mixin is pure — it ships only check helpers, never `__post_init__` or `validate`.** Keeping the *when* (construction-time validation) out of the mixin means the entry-point rule is documented and enforced in each family base, not buried in shared mixin source. The mixin answers only "how do I assert this single constraint", leaving "when do checks run" and "which checks exist" to the config tree.

- **Helpers are `@staticmethod`, called through `self`.** A config invokes a helper through `self` for readability, but the helpers hold no instance state and depend on nothing from the config. This keeps each helper a self-contained predicate-plus-message and lets a config mix several into one `validate_<group>` without coupling.

- **Checks encode only what static analysis and the loader cannot.** Numeric bounds, monotonicity, length, and cross-field relationships are the mixin's domain; type membership, `Literal` sets, and tensor shape / dtype are not — those are the annotation's job and the loader's job. The helper set is intentionally narrow for this reason and grows only when a genuinely new class of runtime constraint appears.

- **Failures raise immediately with a uniform message shape.** Each helper raises `ValueError` carrying the offending field name and value, rather than collecting failures into a report. The first violated constraint stops construction, so a config can never come into existence in a partially-valid state.

## Contracts & invariants

- **Every frozen config inherits `ValidateMixin`** (directly or transitively via `CircuitConfig`) and reaches the helpers through `self`. The mixin contributes no fields and no construction hook.

- **Every config is validated at construction.** A `__post_init__` calls `self.validate()` and nothing else; this entry point sits on the leaf config or is inherited from a family base (e.g. `TIAConfig`), a per-family simplicity choice. `validate()` dispatches the config's `validate_<group>()` checks. A subclass that extends a base whose `validate()` already carries checks opens with `super().validate()` — the only way validation chains up the inheritance tree; a config that adds no checks of its own does not override `validate()`, since the inherited chain already runs.

- **Helpers raise `ValueError` on violation and return `None` on success.** Each takes the value(s) under test plus a human-readable `name` used in the error message. A passing check is silent; there is no success return value to inspect.

- **Provided checks (developer-facing contract):**
  - require a scalar be strictly positive (`value > 0`);
  - require a scalar be non-negative (`value >= 0`);
  - require a sequence be strictly increasing;
  - require a sequence be strictly decreasing;
  - require a sequence meet a minimum length.

  Sequence checks accept any iterable of `int | float`; the monotonicity checks compare consecutive elements pairwise, so an empty or single-element sequence trivially passes.

- **Checks are grouped into `validate_<group>()` methods.** Multi-field configs split into one `validate_<group>()` per logical field group; the standard name `validate_ppa()` covers the area / leakage / latency trio and is supplied by `CircuitConfig`. A single-field config may inline its check in `validate()`. The grouping is not contributed by the mixin — the check helpers are its only surface; the `validate` / `validate_<group>` structure lives in each config and family base.

- **Do not re-validate inner distribution sub-configs.** A config holding a probability-distribution sub-config (`StuckAtFaultConfig`, `TelegraphConfig`, `LognormalConfig`, `GammaConfig`, and the `StateDependent*` variants in `neurox/common/nonideality.py`) must not re-run the helpers against the sub-config's fields in its own `validate*`; the inner config's `__post_init__` already validated on construction. Re-validating duplicates the rule and rots when the inner config changes.

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
- **Tests**: TODO — no dedicated `ValidateMixin` / orchestration test; config `validate*` failures are covered indirectly by subsystem config tests (e.g. `tests/test_adc_family.py`, `tests/test_signal_chain.py`).
- **Decisions**: N/A — no ADR governs the validation mechanism.
