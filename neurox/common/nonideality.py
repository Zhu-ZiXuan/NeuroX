"""Non-ideality models for NeuroX device and circuit simulation.

Each non-ideality is described by a frozen dataclass config so device /
circuit classes pass a single typed object rather than loose scalar
parameters.  Each ``apply_*`` function is a branch-free elementwise
expression decorated with ``@torch.compile`` (via the caller) so random
draws and arithmetic fuse into a single kernel.  This is critical for
broadcast per-cell tensors: eager execution would allocate ~10
full-shape intermediates and OOM at AlexNet conv1 scale.
``torch._dynamo.config.cache_size_limit`` is raised to 128 to prevent
dynamo fallback across the diverse cell-tensor shapes in a typical
multi-layer model.

Most ``apply_*`` functions accept ``config=None`` and return the input
unchanged in that case, so the "is this non-ideality configured?" check
can be folded out of device / circuit code.  The lone exception is
:func:`apply_gaussian` — kept deliberately minimal (a one-line
``x + randn * sigma``) because it is hot enough to be worth fusing
under ``@torch.compile``; callers handle the ``None`` config check
themselves and pass ``sigma`` directly.

State-independent vs state-dependent
------------------------------------

Two flavours of every additive / multiplicative noise:

* **State-independent** — sigma is a config constant; the same
  distribution is sampled at every element.  Use for "global" noise
  whose magnitude does not track signal magnitude (eg. comparator
  thermal noise, stuck-at faults).
* **State-dependent** — sigma is derived per-element from the input
  tensor (or from auxiliary tensors passed alongside).  Use for noise
  whose magnitude grows with conductance state (programming variability,
  retention drift) or with capacitor area (Pelgrom mismatch, kT/C
  sampling noise).

Supported non-ideality types
----------------------------
- ``StuckAtFaultConfig`` / ``apply_stuck_at_fault``: hard fault, cells
  permanently stuck at minimum or maximum conductance.
- ``GaussianConfig`` / ``apply_gaussian``: additive isotropic Gaussian
  noise (state-independent).
- ``StateDependentGaussianConfig`` / ``apply_state_dependent_gaussian``:
  additive Gaussian with sigma linearly proportional to ``|x|``.
- ``LognormalConfig`` / ``apply_lognormal``: multiplicative log-normal
  noise (state-independent).
- ``StateDependentLognormalConfig`` / ``apply_state_dependent_lognormal``:
  multiplicative log-normal with sigma depending on normalised
  conductance.
- ``GammaConfig`` / ``apply_gamma_noise``: multiplicative Gamma noise
  with constant shape and scale, normalised to unit mean
  (state-independent).
- ``StateDependentGammaConfig`` / ``apply_state_dependent_gamma``:
  multiplicative Gamma noise with state-dependent shape parameter.
- ``TelegraphConfig`` / ``apply_telegraph_noise``: random telegraph
  noise (RTN) modelled as a stochastic binary perturbation with
  Gaussian amplitude (state-independent).
- ``apply_pelgrom_mismatch``: Pelgrom-law matching variation on
  binary-weighted unit-cell structures (CDAC capacitors, current-mirror
  DACs).  σ_k ∝ sqrt(C_k / C_unit) — a state-dependent Gaussian whose
  sigma scales with cell area.  Takes the per-unit-cell relative sigma
  ``σ_u`` directly (or ``None`` to skip); no wrapper config dataclass.
- ``apply_lsb_jitter``: integer LSB stochastic-rounding jitter on
  digital ADC codes (training-mode coarse fallback).

Specialised noise sources (eg. kT/C sampling noise, where σ depends on
runtime state — sampling capacitance, temperature) are implemented at
the call site by computing σ inline and piping through
:func:`apply_gaussian`.  No dedicated helper is provided.
"""

from dataclasses import dataclass

import torch
import torch._dynamo
from torch import Tensor

# ---------------------------------------------------------------------------
# Stuck-at fault
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StuckAtFaultConfig:
    """Stuck-at fault probabilities.

    Attributes:
        p_at_min: Probability that a cell is stuck at the minimum value.
        p_at_max: Probability that a cell is stuck at the maximum value.
    """

    p_at_min: float
    p_at_max: float

    def validate(self) -> None:
        """Validate fault probabilities."""
        if (self.p_at_min < 0.0) or (self.p_at_max < 0.0) or (self.p_at_min + self.p_at_max >= 1.0):
            raise ValueError(f"Stuck-at fault probabilities not applicable: {self}")


def apply_stuck_at_fault(x: Tensor, config: StuckAtFaultConfig | None, min_val: float, max_val: float) -> Tensor:
    """Replace cells with stuck-at-min / stuck-at-max values.

    Args:
        x: Input conductance. Shape: arbitrary.
        config: Stuck-at fault probabilities, or ``None`` to skip.
        min_val: Stuck-at-min replacement value.
        max_val: Stuck-at-max replacement value.

    Returns:
        Conductance with stuck-at faults applied. Shape: same as ``x``.
    """
    if config is None:
        return x
    rand_mask = torch.rand_like(x)
    p_min = config.p_at_min
    p_both = config.p_at_min + config.p_at_max
    is_min = rand_mask < p_min
    is_max = (rand_mask >= p_min) & (rand_mask < p_both)
    return torch.where(is_min, min_val, torch.where(is_max, max_val, x))


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------


def apply_gaussian(x: Tensor, sigma: float | Tensor) -> Tensor:
    """Apply additive Gaussian noise.

    Args:
        x: Input tensor..
        sigma: Standard deviation of the additive noise.

    Returns:
        Noisy tensor.  Shape: same as ``x``.
    """
    return x + torch.randn_like(x) * sigma


@dataclass(frozen=True)
class StateDependentGaussianConfig:
    """State-dependent Gaussian noise config.

    Attributes:
        sigma_slope: Sigma slope vs ``|x|``.
        sigma_intercept: Base sigma at ``|x| = 0``.
    """

    sigma_slope: float
    sigma_intercept: float


def apply_state_dependent_gaussian(x: Tensor, config: StateDependentGaussianConfig | None) -> Tensor:
    """Apply Gaussian noise whose sigma scales with the magnitude of ``x``.

    Args:
        x: Input conductance. Shape: arbitrary.
        config: Slope and intercept of the per-element sigma, or ``None``
            to skip.

    Returns:
        Noisy tensor. Shape: same as ``x``.
    """
    if config is None:
        return x
    sigma = config.sigma_slope * x.abs() + config.sigma_intercept
    return x + torch.randn_like(x) * sigma


# ---------------------------------------------------------------------------
# Log-normal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LognormalConfig:
    """Multiplicative log-normal noise config.

    Attributes:
        sigma: Underlying normal sigma.
    """

    sigma: float


def apply_lognormal(x: Tensor, config: LognormalConfig | None) -> Tensor:
    """Apply multiplicative log-normal noise.

    Args:
        x: Input conductance. Shape: arbitrary.
        config: Log-normal sigma, or ``None`` to skip.

    Returns:
        Noisy tensor. Shape: same as ``x``.
    """
    if config is None:
        return x
    return x * torch.exp(torch.randn_like(x) * config.sigma)


@dataclass(frozen=True)
class StateDependentLognormalConfig:
    """State-dependent log-normal noise config.

    Attributes:
        sigma_slope: Slope of sigma vs normalised state.
        sigma_intercept: Base sigma at min state.
        min_val: Min value used for normalisation.
        max_val: Max value used for normalisation.
    """

    sigma_slope: float
    sigma_intercept: float
    min_val: float
    max_val: float


def apply_state_dependent_lognormal(x: Tensor, config: StateDependentLognormalConfig | None) -> Tensor:
    """Apply log-normal noise whose sigma depends on normalised state.

    Args:
        x: Input conductance. Shape: arbitrary.
        config: State-dependent sigma config, or ``None`` to skip.

    Returns:
        Noisy tensor. Shape: same as ``x``.
    """
    if config is None:
        return x
    x_norm = (x - config.min_val) / (config.max_val - config.min_val + 1e-12)
    sigma = x_norm * (-config.sigma_slope) + config.sigma_intercept
    return x * torch.exp(torch.randn_like(x) * sigma)


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GammaConfig:
    """Multiplicative Gamma noise config (constant shape and scale).

    Attributes:
        shape_k: Gamma shape parameter ``k``.
        scale_theta: Gamma scale parameter ``theta``.
    """

    shape_k: float
    scale_theta: float


def apply_gamma_noise(x: Tensor, config: GammaConfig | None) -> Tensor:
    """Apply multiplicative Gamma noise normalised to unit mean.

    Args:
        x: Input tensor. Shape: arbitrary.
        config: Constant Gamma config, or ``None`` to skip.

    Returns:
        Noisy tensor. Shape: same as ``x``.
    """
    if config is None:
        return x
    gamma_dist = torch.distributions.Gamma(config.shape_k, 1.0 / config.scale_theta)
    gamma_sample = gamma_dist.sample(x.shape).to(x.device, x.dtype)
    mean = config.shape_k * config.scale_theta
    return x * (gamma_sample / mean)


@dataclass(frozen=True)
class StateDependentGammaConfig:
    """State-dependent Gamma noise config.

    Attributes:
        k_slope: Shape slope vs normalised state.
        k_intercept: Base shape value.
        theta: Shared scale parameter.
        min_val: Min value used for normalisation.
        max_val: Max value used for normalisation.
    """

    k_slope: float
    k_intercept: float
    theta: float
    min_val: float
    max_val: float


def apply_state_dependent_gamma(x: Tensor, config: StateDependentGammaConfig | None) -> Tensor:
    """Apply state-dependent Gamma noise normalised to unit mean.

    PyTorch's gamma sampler requires float32 or higher; the function
    transparently casts low-precision inputs (e.g. bfloat16) to float32
    for the sampling and casts the result back at the end.

    Args:
        x: Input conductance. Shape: arbitrary.
        config: State-dependent Gamma config, or ``None`` to skip.

    Returns:
        Noisy tensor. Shape: same as ``x``.
    """
    if config is None:
        return x
    in_dtype = x.dtype
    needs_cast = in_dtype not in (torch.float32, torch.float64)
    x32 = x.float() if needs_cast else x

    # --- normalise conductance state ---
    x_norm = (x32 - config.min_val) / (config.max_val - config.min_val + 1e-12)

    # --- derive state-dependent shape ---
    k = (x_norm * config.k_slope + config.k_intercept).clamp(min=0.1)

    # --- sample gamma noise elementwise ---
    theta_tensor = torch.full(x32.shape, config.theta, dtype=x32.dtype, device=x32.device)
    rate = 1.0 / theta_tensor
    gamma_sample = torch.distributions.Gamma(concentration=k, rate=rate).sample()

    # --- normalise to unit-mean gain ---
    mean = (k * theta_tensor).clamp(min=1e-12)
    result = x32 * (gamma_sample / mean)
    return result.to(in_dtype) if needs_cast else result


# ---------------------------------------------------------------------------
# Telegraph (Random Telegraph Noise)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TelegraphConfig:
    """Random telegraph noise config.

    Attributes:
        amplitude_mean: Mean amplitude of the perturbation.
        amplitude_std: Std of the amplitude.
        p_high_state: Probability that a cell is in the high RTN state.
    """

    amplitude_mean: float
    amplitude_std: float
    p_high_state: float


def apply_telegraph_noise(x: Tensor, config: TelegraphConfig | None) -> Tensor:
    """Apply random telegraph noise.

    Composed branch-free as ``perturb = amplitude * sign * mask`` so a
    higher-level ``@torch.compile`` caller (e.g. ``RRAM.read_g__mS``)
    fuses the three random draws and the float-promoted ``torch.where``
    intermediate into a single elementwise kernel.

    Args:
        x: Input conductance. Shape: arbitrary.
        config: Random telegraph noise config, or ``None`` to skip.

    Returns:
        Noisy tensor. Shape: same as ``x``.
    """
    if config is None:
        return x
    amplitude = torch.randn_like(x) * config.amplitude_std + config.amplitude_mean
    sign = torch.where(torch.rand_like(x) < 0.5, -1.0, 1.0).to(dtype=x.dtype)
    mask = (torch.rand_like(x) < config.p_high_state).to(dtype=x.dtype)
    return x + amplitude * sign * mask


# ---------------------------------------------------------------------------
# Pelgrom-law mismatch (state-dependent area-scaled Gaussian)
# ---------------------------------------------------------------------------


def apply_pelgrom_mismatch(
    ideal: Tensor,
    sigma_relative: float | None,
    *,
    unit: float,
    floor: float | None = None,
) -> Tensor:
    """Add Pelgrom-scaled Gaussian mismatch to a binary-weighted ladder.

    A binary-weighted device of value ``X_k = N_k · X_unit`` (built by
    paralleling ``N_k`` unit cells of value ``X_unit``) inherits a
    matching sigma

        σ_k = sqrt(N_k) · σ_u · X_unit

    where ``σ_u`` is the per-unit-cell relative sigma (the matching
    parameter quoted by the foundry).  Used for CDAC capacitor
    mismatch (``X = C``), current-mirror DACs (``X = I``), etc.

    Args:
        ideal: Tensor of nominal per-cell values (eg. cap weights in fF).
            Shape: arbitrary; trailing dim typically indexes the
            binary-weighted ladder.
        sigma_relative: Per-unit-cell relative sigma ``σ_u`` (eg.
            ``0.01`` for 1% matching), or ``None`` to skip.
        unit: Single-unit-cell value ``X_unit`` (same units as
            ``ideal``).  Used to convert cell value → number of unit
            cells fused.
        floor: Optional minimum clamp applied after sampling so the
            output stays physical (eg. positive caps).

    Returns:
        Tensor with the same shape / dtype / device as ``ideal``.
    """
    if sigma_relative is None or sigma_relative <= 0.0:
        return ideal
    sigma = torch.sqrt(ideal / unit) * (sigma_relative * unit)
    out = ideal + torch.randn_like(ideal) * sigma
    if floor is not None:
        out = out.clamp_min(floor)
    return out


# ---------------------------------------------------------------------------
# LSB stochastic-rounding jitter on integer codes
# ---------------------------------------------------------------------------


def apply_lsb_jitter(
    code: Tensor,
    *,
    n_bits: int,
    enabled: bool,
) -> Tensor:
    """Add a Bernoulli(0.5) ±0/+1 LSB jitter to an integer code.

    Coarse stochastic-rounding fallback for ADCs whose physical model
    does not already inject per-cycle randomness.  Output is clamped
    to ``[0, 2 ** n_bits - 1]``.

    Args:
        code: Integer code tensor.
        n_bits: Active bit width (used for the upper clamp).
        enabled: Master toggle.  ``False`` returns ``code`` unchanged.

    Returns:
        Jittered code with the same dtype / device as ``code``.
    """
    if not enabled:
        return code
    max_code = (1 << n_bits) - 1
    jitter = torch.randint(low=0, high=2, size=code.shape, dtype=code.dtype, device=code.device)
    return (code + jitter).clamp(min=0, max=max_code)
