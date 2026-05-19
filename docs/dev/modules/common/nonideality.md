# `neurox/common/nonideality.py`

## Current role

`nonideality.py` collects the reusable analog-noise / mismatch kernels. Each kernel is a small `torch.compile`-friendly function with a narrow interface, imported by name.

## State-independent vs state-dependent

Every additive / multiplicative noise comes in two flavours:

- **State-independent** — sigma is a config constant; the same distribution is sampled at every element. Use for noise whose magnitude does not track signal magnitude (comparator thermal noise, stuck-at faults).
- **State-dependent** — sigma is derived per-element from the input tensor (or auxiliary tensors). Use for noise whose magnitude grows with conductance state (programming variability, retention drift) or with cell area (Pelgrom mismatch, kT/C sampling).

## Provided primitives

- `apply_stuck_at_fault(tensor, cfg)` — Bernoulli stuck-at fault parameterised by `StuckAtFaultConfig`.
- `apply_gaussian(tensor, sigma)` — additive isotropic Gaussian draw (state-independent). Takes `sigma` directly so it inlines tightly under `@torch.compile`; the `sigma is None` guard is handled at the call site.
- `apply_state_dependent_gaussian(tensor, cfg)` — additive Gaussian with sigma proportional to `|x|`.
- `apply_lognormal(tensor, cfg)` — multiplicative log-normal.
- `apply_state_dependent_lognormal(tensor, cfg)` — multiplicative log-normal with sigma depending on normalised conductance.
- `apply_gamma_noise(tensor, cfg)` — multiplicative Gamma noise normalised to unit mean.
- `apply_state_dependent_gamma(tensor, cfg)` — Gamma noise with state-dependent shape parameter.
- `apply_telegraph_noise(tensor, cfg)` — RTN-style binary-state perturbation with Gaussian amplitude.
- `apply_pelgrom_mismatch(tensor, sigma_relative, *, unit, floor)` — Pelgrom-area-scaled multiplicative mismatch (σ_k ∝ √(C_k / C_unit)) with a positive floor. Takes `sigma_relative` directly, no wrapper config.
- `apply_lsb_jitter(tensor, lsb)` — uniform LSB-jitter for stochastic rounding.

The associated config dataclasses (`TelegraphConfig`, `StateDependentGammaConfig`, `StuckAtFaultConfig`, …) live in this file as the single canonical set of spec dataclasses for the kernels above.

Specialised noise sources whose σ depends on runtime state (kT/C sampling: σ = √(kT/C) with C changing per call) are implemented at the call site by computing σ inline and piping through `apply_gaussian` — no dedicated helper is provided.

## Why these live in common

The kernels are domain-agnostic. Centralising them gives one place to optimise the dynamo trace and one place to audit the noise math. `torch._dynamo.config.cache_size_limit` is raised to 128 here to prevent dynamo fallback across the diverse cell-tensor shapes seen across a typical multi-layer model.
