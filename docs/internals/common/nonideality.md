# Non-ideality Kernels — Implementation

## Summary

`neurox/common/nonideality.py` is the single canonical home for the analog noise and mismatch kernels shared across the codebase, together with the frozen config dataclasses that parameterise them. Every kernel is a small, stateless `apply_*` function with a narrow signature: it takes the source value (a scalar sigma, or a config dataclass) plus a kw-only `enabled: bool`, and returns the input tensor perturbed when enabled or unchanged when disabled. The owning module gates each kernel by passing `enabled=self.policy.<source>` straight from its nonideality policy, so the call site is unconditional — no `None` check, no caller `if`-gate. The kernels are domain-agnostic math; the physics taxonomy (state-independent vs state-dependent, the Pelgrom area law, the per-source toggle convention) is the shared spec in [notation_conventions](../../reference/notation_conventions.md) and is not restated here.

## Design decisions

- **Toggle inside the kernel, not at the call site.** Each `apply_*` takes `*, enabled: bool` and short-circuits to the unchanged input when false. The choice moves the on/off branch out of every consumer: the owning module wires `enabled=self.policy.<source>` once and the noise call is written exactly once, with no conditional around it. This keeps consumer code branch-free and makes the policy the single source of truth for which sources are active.
- **No `None` sentinel, no caller gate.** A disabled source is expressed by `enabled=False`, never by passing `None` and skipping the call. There is deliberately no overload where the config or sigma may be absent — the value is always supplied and the boolean alone decides whether it is used. This removes a class of "forgot to gate" bugs and keeps the policy boolean as the only knob.
- **Sigma-direct vs config-object split.** Kernels whose spread is a single scalar take that scalar directly (`apply_gaussian` takes `sigma`, `apply_pelgrom_mismatch` takes `sigma_relative`) so the value inlines tightly under `@torch.compile`. Kernels with several coupled parameters take a frozen config dataclass (e.g. `TelegraphConfig`, `LognormalConfig`) so the parameter set stays validated and named.
- **Configs are colocated with their kernels.** The spec dataclasses (`StuckAtFaultConfig`, `StateDependentGaussianConfig`, `LognormalConfig`, `StateDependentLognormalConfig`, `GammaConfig`, `StateDependentGammaConfig`, `TelegraphConfig`) live in this file as the single canonical definition, rather than being scattered across consumer modules, so the noise math and its parameters are audited in one place.
- **Runtime-state sigma is computed at the call site.** A source whose sigma depends on per-call runtime state (kT/C sampling, where sigma = sqrt(kT/C) with C varying per call) gets no dedicated kernel: the consumer computes sigma inline and pipes it through `apply_gaussian`. Only fixed-config and input-derived sources earn a named helper.

## Contracts & invariants

- **Disabled is identity.** With `enabled=False` every `apply_*` returns the input tensor unchanged (same object, no draw). This is the defining contract that lets consumers call unconditionally.
- **Shape / dtype / device preserved.** With `enabled=True` the output has the same shape, dtype, and device as the input tensor; the kernels draw their randomness `*_like` the input.
- **`enabled` is kw-only.** Across the whole family `enabled` is passed by keyword (after `*`), so it can never be confused positionally with a sigma or config argument.
- **The kernel family:**
  - `apply_stuck_at_fault(tensor, config, min_val, max_val, *, enabled)` — Bernoulli stuck-at fault from `StuckAtFaultConfig`, clamped into `[min_val, max_val]`.
  - `apply_gaussian(tensor, sigma, *, enabled)` — additive isotropic Gaussian; `sigma` is a scalar or tensor passed directly (state-independent).
  - `apply_state_dependent_gaussian(tensor, config, *, enabled)` — additive Gaussian with per-element sigma proportional to the input magnitude.
  - `apply_lognormal(tensor, config, *, enabled)` — multiplicative log-normal.
  - `apply_state_dependent_lognormal(tensor, config, *, enabled)` — multiplicative log-normal with sigma depending on normalised conductance.
  - `apply_gamma_noise(tensor, config, *, enabled)` — multiplicative Gamma noise normalised to unit mean.
  - `apply_state_dependent_gamma(tensor, config, *, enabled)` — Gamma noise with a state-dependent shape parameter.
  - `apply_telegraph_noise(tensor, config, *, enabled)` — RTN-style binary-state perturbation with Gaussian amplitude.
  - `apply_pelgrom_mismatch(ideal, sigma_relative, *, unit, floor=None, enabled)` — Pelgrom area-scaled multiplicative mismatch; per-cell sigma grows as sqrt(cell / unit); an optional `floor` clamps the perturbed result from below.
  - `apply_lsb_jitter(code, *, unsigned_max, enabled)` — Bernoulli +0/+1 one-LSB jitter on an integer code, clamped to `[0, unsigned_max]`.
- **`apply_lsb_jitter` self-clamps.** The output is clamped to `[0, unsigned_max]`, absorbing the +1 overflow at the top of the legal range; the caller does not add a second clamp. `unsigned_max` is the precomputed `2 ** n_bits - 1`.
- **`apply_pelgrom_mismatch` floor is optional.** `floor=None` leaves the sampled output unclamped; a positive `floor` applies a lower clamp after sampling so the value never collapses to zero.

## Performance & resources

- **Stateless and allocation-light.** No persistent buffers; the only allocations are the per-call noise draws, sampled on the input's device and dtype. Disabled kernels allocate nothing.
- **`enabled` short-circuit is trace-time, not value-dependent.** On the [compiled consumer path](../compile/contracts.md) `enabled` is a Python `bool` resolved at trace time, so the disabled short-circuit is a trace-time branch, not tensor-value control flow — dynamo specialises each kernel on the boolean. The sigma-direct kernels inline their scalar so the draw fuses with the consumer graph.
- **`apply_lsb_jitter` avoids a SymInt shift.** `unsigned_max` is passed as a precomputed Python int rather than computed as `1 << n_bits` inside the kernel, so the compiled graph never contains a SymInt left-shift op.

## Gotchas

- **Disabling means `enabled=False`, never "don't call".** Wrapping the call in a caller-side `if` defeats the design and reintroduces the branch the toggle exists to remove; always call unconditionally and pass the policy boolean.
- **State-independent vs state-dependent is not interchangeable.** Choosing the wrong flavour (a fixed sigma where the spread should track signal magnitude, or vice versa) silently mis-models the noise rather than erroring. Pick the kernel that matches the physical source per the taxonomy in [notation_conventions](../../reference/notation_conventions.md).
- **`apply_pelgrom_mismatch` takes `unit` in the same units as `ideal`.** A mismatched `unit` rescales the whole sqrt(cell / unit) ladder silently; both must be in the same physical units.

## Known limitations

- **No kT/C kernel.** Runtime-state sigma sources (kT/C sampling) have no dedicated helper and are implemented at the call site through `apply_gaussian`; a generic runtime-sigma helper is not provided.
- **No correlated / spatially structured noise.** Every draw is independent per element. Sources with spatial correlation (e.g. systematic process gradients across an array) are not modelled by this family.

---

- **Reference**: [notation_conventions](../../reference/notation_conventions.md) — noise taxonomy, per-source toggle, Pelgrom area law.
- **Implementation**: `neurox/common/nonideality.py`
- **Tests**: TODO — no dedicated test module for the kernel family yet.
- **Decisions**: N/A.
