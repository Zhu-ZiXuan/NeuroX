# Non-ideality kernels

`neurox/primitive/nonideality.py` is the single canonical home for the analog noise and mismatch kernels shared across the codebase, together with the frozen config dataclasses that parameterise them. Each kernel is a small, stateless `apply_*` function that takes the source value (a scalar sigma, or a config dataclass) plus a keyword-only `enabled: bool`, and returns the input perturbed when enabled or unchanged when disabled. The noise math itself and the physical taxonomy (state-independent vs state-dependent, the Pelgrom area law) are the science spec and are not restated here.

## Design decisions

- **Each kernel owns its enable branch.** Every `apply_*` function takes `*, enabled: bool` and returns the input unchanged when false. Disabled sources are represented only by `enabled=False`; the API has no `None` overload. Because `enabled` is a plain Python `bool`, the branch resolves at trace time rather than becoming tensor control flow, and so stays [dynamo-safe](../compile/contracts.md).
- **Sigma-direct vs config-object split.** A kernel whose spread is a single scalar takes that scalar directly (`apply_gaussian`, `apply_relative_gaussian`, `apply_pelgrom_mismatch`) so the value inlines tightly under compilation; a kernel with several coupled parameters takes a frozen config dataclass so the parameter set stays named and validated in one construction.
- **Absolute and relative spread are separate kernels.** `apply_gaussian` adds an absolute sigma in the signal's own unit; `apply_relative_gaussian` scales by `1 + N(0, sigma_relative)`, so the sigma is dimensionless and an exact zero stays exactly zero. A source whose spread is quoted as a percentage of its nominal value uses the relative kernel.
- **A disabled kernel returns the input object itself.** `apply_relative_gaussian(x, ..., enabled=False)` returns `x`, not a clone, so a wide broadcast view stays unmaterialized when the toggle is off. A caller must therefore treat the result as read-only and never mutate it in place.
- **Configs are colocated with their kernels.** The spec dataclasses live in this file as their single canonical definition, so the noise math and its parameters can be audited together.
- **No dedicated kernel for externally derived sigma.** A source whose sigma depends on per-call runtime state uses `apply_gaussian` after its sigma has been calculated. Named helpers are reserved for fixed-config and input-derived sources.
- **Kernels do not infer state classification.** The selected sigma path encodes whether a source is state-independent or state-dependent. Kernels trust the supplied value and do not validate that classification, so a wrong classification silently produces a wrong model.

## Known limitations

- **Independent draws only.** Every kernel draws its randomness independently per element. Sources with spatial correlation (e.g. systematic process gradients across an array) are not modelled by this family.
- **Pelgrom mismatch sigma direction.** The per-cell mismatch sigma in `apply_pelgrom_mismatch` grows with cell area, an absolute-spread framing; whether that is the intended framing is unconfirmed. TODO — domain author to confirm the sigma direction.

---

- **Reference**: [reference/nonideality](../../reference/primitive/nonideality.md) — noise taxonomy, Pelgrom area law.
- **Implementation**: `neurox/primitive/nonideality.py`
- **Tests**: TODO — no dedicated test module for the kernel family yet.
