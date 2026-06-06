# `neurox/common/nonideality.py`

## Current role

`nonideality.py` collects the reusable analog-noise / mismatch kernels. Each kernel is a small `torch.compile`-friendly function with a narrow interface, imported by name.

## State-independent vs state-dependent

Every additive / multiplicative noise comes in two flavours:

- **State-independent** — sigma is a config constant; the same distribution is sampled at every element. Use for noise whose magnitude does not track signal magnitude (comparator thermal noise, stuck-at faults).
- **State-dependent** — sigma is derived per-element from the input tensor (or auxiliary tensors). Use for noise whose magnitude grows with conductance state (programming variability, retention drift) or with cell area (Pelgrom mismatch, kT/C sampling).

## Toggle contract

Every `apply_*` helper takes a `*, enabled: bool` kw-only parameter and returns the input unchanged when `enabled=False`. Callers pass `enabled=self.policy.<source>` straight from the owning module's nonideality policy; no helper checks for `None` and no caller writes an `if`-gate.

## Provided primitives

- `apply_stuck_at_fault(tensor, config, min_val, max_val, *, enabled)` — Bernoulli stuck-at fault parameterised by `StuckAtFaultConfig`.
- `apply_gaussian(tensor, sigma, *, enabled)` — additive isotropic Gaussian draw (state-independent). Takes `sigma` directly so it inlines tightly under `@torch.compile`.
- `apply_state_dependent_gaussian(tensor, config, *, enabled)` — additive Gaussian with sigma proportional to `|x|`.
- `apply_lognormal(tensor, config, *, enabled)` — multiplicative log-normal.
- `apply_state_dependent_lognormal(tensor, config, *, enabled)` — multiplicative log-normal with sigma depending on normalised conductance.
- `apply_gamma_noise(tensor, config, *, enabled)` — multiplicative Gamma noise normalised to unit mean.
- `apply_state_dependent_gamma(tensor, config, *, enabled)` — Gamma noise with state-dependent shape parameter.
- `apply_telegraph_noise(tensor, config, *, enabled)` — RTN-style binary-state perturbation with Gaussian amplitude.
- `apply_pelgrom_mismatch(tensor, sigma_relative, *, unit, floor, enabled)` — Pelgrom-area-scaled multiplicative mismatch (σ_k ∝ √(C_k / C_unit)) with a positive floor. Takes `sigma_relative` directly, no wrapper config.
- `apply_lsb_jitter(tensor, *, n_bits, enabled)` — uniform LSB-jitter for stochastic rounding.

The associated config dataclasses (`TelegraphConfig`, `StateDependentGammaConfig`, `StuckAtFaultConfig`, …) live in this file as the single canonical set of spec dataclasses for the kernels above.

Specialised noise sources whose σ depends on runtime state (kT/C sampling: σ = √(kT/C) with C changing per call) are implemented at the call site by computing σ inline and piping through `apply_gaussian` — no dedicated helper is provided.

## Why these live in common

The kernels are domain-agnostic. Centralising them gives one place to optimise the dynamo trace and one place to audit the noise math. `torch._dynamo.config.cache_size_limit` is raised to 128 here to prevent dynamo fallback across the diverse cell-tensor shapes seen across a typical multi-layer model.
