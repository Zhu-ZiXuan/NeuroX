# Non-ideality kernels

## Summary

`neurox/primitive/nonideality.py` is the single canonical home for the analog noise and mismatch kernels shared across the codebase, together with the frozen config dataclasses that parameterise them. Each kernel is a small, stateless `apply_*` function that takes the source value (a scalar sigma, or a config dataclass) plus a keyword-only `enabled: bool`, and returns the input perturbed when enabled or unchanged when disabled. The noise math itself and the physical taxonomy (state-independent vs state-dependent, the Pelgrom area law) are the science spec and are not restated here.

## Design decisions

- **Toggle inside the kernel, not at the call site.** Each `apply_*` takes `*, enabled: bool` and short-circuits to the unchanged input when false, so every non-ideality is switched on or off individually. This moves the on/off branch out of every consumer: the caller wires `enabled=<policy flag>` once and writes the noise call exactly once with no conditional around it, keeping consumer code branch-free and making the policy the single source of truth for which sources are active. A disabled source is always expressed as `enabled=False`, never by passing `None` or skipping the call — there is deliberately no absent-value overload, which removes a class of "forgot to gate" bugs. Because `enabled` is a plain Python `bool`, the short-circuit resolves at trace time rather than becoming tensor control flow, and so stays [dynamo-safe](../compile/contracts.md).
- **Sigma-direct vs config-object split.** A kernel whose spread is a single scalar takes that scalar directly (`apply_gaussian`, `apply_pelgrom_mismatch`) so the value inlines tightly under compilation; a kernel with several coupled parameters takes a frozen config dataclass so the parameter set stays named and validated in one construction.
- **Configs colocated with their kernels.** The spec dataclasses live in this file as their single canonical definition rather than being scattered across consumer modules, so the noise math and its parameters are audited in one place.
- **Runtime-state sigma is computed at the call site.** A source whose sigma depends on per-call runtime state (charge-sampling noise, whose sigma follows a capacitance that varies per call) gets no dedicated kernel: the caller computes sigma inline and pipes it through `apply_gaussian`. Only fixed-config and input-derived sources earn a named helper.
- **The caller owns the state classification.** Which sigma path a source takes — a fixed scalar, an area- or state-derived sigma, or a call-site sigma — encodes whether it is state-independent or state-dependent; a kernel trusts that choice and never validates it, so a misclassification silently mis-models the source rather than raising.

## Known limitations

- **Independent draws only.** Every kernel draws its randomness independently per element. Sources with spatial correlation (e.g. systematic process gradients across an array) are not modelled by this family.
- **Pelgrom mismatch sigma direction.** The per-cell mismatch sigma in `apply_pelgrom_mismatch` grows with cell area, an absolute-spread framing; whether that is the intended framing is unconfirmed. TODO — domain author to confirm the sigma direction.

---

- **Reference**: [reference/nonideality](../../reference/primitive/nonideality.md) — noise taxonomy, Pelgrom area law.
- **Implementation**: `neurox/primitive/nonideality.py`
- **Tests**: TODO — no dedicated test module for the kernel family yet.
