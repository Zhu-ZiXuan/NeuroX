"""Reusable analog non-ideality kernels and their config dataclasses.

See also:
    docs/dev/modules/common/nonideality.md
    docs/dev/architecture/noise_and_toggles.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.mixin import ValidateMixin

# ---------------------------------------------------------------------------
# Stuck-at fault
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StuckAtFaultConfig(ValidateMixin):
    """Stuck-at fault probabilities.

    Attributes:
        p_at_min: Probability that a cell is stuck at the minimum value.
        p_at_max: Probability that a cell is stuck at the maximum value.
    """

    p_at_min: float
    p_at_max: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._require_nonneg(self.p_at_min, "p_at_min")
        self._require_nonneg(self.p_at_max, "p_at_max")
        if not (self.p_at_min + self.p_at_max < 1.0):
            raise ValueError(f"require: p_at_min ({self.p_at_min}) + p_at_max ({self.p_at_max}) < 1")


def apply_stuck_at_fault(
    x: Tensor,
    config: StuckAtFaultConfig,
    min_val: float,
    max_val: float,
    *,
    enabled: bool,
) -> Tensor:
    """Replace cells with stuck-at-min / stuck-at-max values.

    Args:
        x: Input conductance. Shape: arbitrary.
        config: Stuck-at fault probabilities.
        min_val: Stuck-at-min replacement value.
        max_val: Stuck-at-max replacement value.
        enabled: Master toggle. ``False`` returns ``x`` unchanged.

    Returns:
        Conductance with stuck-at faults applied. Shape: same as ``x``.
    """
    if not enabled:
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


def apply_gaussian(x: Tensor, sigma: float | Tensor, *, enabled: bool) -> Tensor:
    """Apply additive Gaussian noise.

    Args:
        x: Input tensor.
        sigma: Standard deviation of the additive noise.
        enabled: Master toggle. ``False`` returns ``x`` unchanged.

    Returns:
        Noisy tensor. Shape: same as ``x``.
    """
    if not enabled:
        return x
    return x + torch.randn_like(x) * sigma


@dataclass(frozen=True)
class StateDependentGaussianConfig(ValidateMixin):
    """State-dependent Gaussian noise config.

    Attributes:
        sigma_slope: Sigma slope vs ``|x|``.
        sigma_intercept: Base sigma at ``|x| = 0``.
    """

    sigma_slope: float
    sigma_intercept: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._require_nonneg(self.sigma_slope, "sigma_slope")
        self._require_nonneg(self.sigma_intercept, "sigma_intercept")


def apply_state_dependent_gaussian(
    x: Tensor,
    config: StateDependentGaussianConfig,
    *,
    enabled: bool,
) -> Tensor:
    """Apply Gaussian noise whose sigma scales with the magnitude of ``x``.

    Args:
        x: Input conductance. Shape: arbitrary.
        config: Slope and intercept of the per-element sigma.
        enabled: Master toggle. ``False`` returns ``x`` unchanged.

    Returns:
        Noisy tensor. Shape: same as ``x``.
    """
    if not enabled:
        return x
    sigma = config.sigma_slope * x.abs() + config.sigma_intercept
    return x + torch.randn_like(x) * sigma


# ---------------------------------------------------------------------------
# Log-normal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LognormalConfig(ValidateMixin):
    """Multiplicative log-normal noise config.

    Attributes:
        sigma: Underlying normal sigma.
    """

    sigma: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._require_nonneg(self.sigma, "sigma")


def apply_lognormal(x: Tensor, config: LognormalConfig, *, enabled: bool) -> Tensor:
    """Apply multiplicative log-normal noise.

    Args:
        x: Input conductance. Shape: arbitrary.
        config: Log-normal sigma.
        enabled: Master toggle. ``False`` returns ``x`` unchanged.

    Returns:
        Noisy tensor. Shape: same as ``x``.
    """
    if not enabled:
        return x
    return x * torch.exp(torch.randn_like(x) * config.sigma)


@dataclass(frozen=True)
class StateDependentLognormalConfig(ValidateMixin):
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

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._require_nonneg(self.sigma_slope, "sigma_slope")
        self._require_nonneg(self.sigma_intercept, "sigma_intercept")
        if not (self.max_val > self.min_val):
            raise ValueError(f"require: max_val ({self.max_val}) > min_val ({self.min_val})")


def apply_state_dependent_lognormal(
    x: Tensor,
    config: StateDependentLognormalConfig,
    *,
    enabled: bool,
) -> Tensor:
    """Apply log-normal noise whose sigma depends on normalised state.

    Args:
        x: Input conductance. Shape: arbitrary.
        config: State-dependent sigma config.
        enabled: Master toggle. ``False`` returns ``x`` unchanged.

    Returns:
        Noisy tensor. Shape: same as ``x``.
    """
    if not enabled:
        return x
    x_norm = (x - config.min_val) / (config.max_val - config.min_val + 1e-12)
    sigma = x_norm * (-config.sigma_slope) + config.sigma_intercept
    return x * torch.exp(torch.randn_like(x) * sigma)


# ---------------------------------------------------------------------------
# Gamma
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GammaConfig(ValidateMixin):
    """Multiplicative Gamma noise config (constant shape and scale).

    Attributes:
        shape_k: Gamma shape parameter ``k``.
        scale_theta: Gamma scale parameter ``theta``.
    """

    shape_k: float
    scale_theta: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._require_pos(self.shape_k, "shape_k")
        self._require_pos(self.scale_theta, "scale_theta")


def apply_gamma_noise(x: Tensor, config: GammaConfig, *, enabled: bool) -> Tensor:
    """Apply multiplicative Gamma noise normalised to unit mean.

    Args:
        x: Input tensor. Shape: arbitrary.
        config: Constant Gamma config.
        enabled: Master toggle. ``False`` returns ``x`` unchanged.

    Returns:
        Noisy tensor. Shape: same as ``x``.
    """
    if not enabled:
        return x
    gamma_dist = torch.distributions.Gamma(config.shape_k, 1.0 / config.scale_theta)
    gamma_sample = gamma_dist.sample(x.shape).to(x.device, x.dtype)
    mean = config.shape_k * config.scale_theta
    return x * (gamma_sample / mean)


@dataclass(frozen=True)
class StateDependentGammaConfig(ValidateMixin):
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

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._require_pos(self.k_intercept, "k_intercept")
        self._require_pos(self.theta, "theta")
        if not (self.max_val > self.min_val):
            raise ValueError(f"require: max_val ({self.max_val}) > min_val ({self.min_val})")


def apply_state_dependent_gamma(
    x: Tensor,
    config: StateDependentGammaConfig,
    *,
    enabled: bool,
) -> Tensor:
    """Apply state-dependent Gamma noise normalised to unit mean.

    PyTorch's gamma sampler requires float32 or higher; the function
    transparently casts low-precision inputs (e.g. bfloat16) to float32
    for the sampling and casts the result back at the end.

    Args:
        x: Input conductance. Shape: arbitrary.
        config: State-dependent Gamma config.
        enabled: Master toggle. ``False`` returns ``x`` unchanged.

    Returns:
        Noisy tensor. Shape: same as ``x``.
    """
    if not enabled:
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
class TelegraphConfig(ValidateMixin):
    """Random telegraph noise config.

    Attributes:
        amplitude_mean: Mean amplitude of the perturbation.
        amplitude_std: Std of the amplitude.
        p_high_state: Probability that a cell is in the high RTN state.
    """

    amplitude_mean: float
    amplitude_std: float
    p_high_state: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self._require_nonneg(self.amplitude_std, "amplitude_std")
        if not (0.0 <= self.p_high_state <= 1.0):
            raise ValueError(f"require: 0 <= p_high_state ({self.p_high_state}) <= 1")


def apply_telegraph_noise(x: Tensor, config: TelegraphConfig, *, enabled: bool) -> Tensor:
    """Apply random telegraph noise.

    Composed branch-free as ``perturb = amplitude * sign * mask`` so
    the surrounding kernel can fuse the random draws efficiently.

    Args:
        x: Input conductance. Shape: arbitrary.
        config: Random telegraph noise config.
        enabled: Master toggle. ``False`` returns ``x`` unchanged.

    Returns:
        Noisy tensor. Shape: same as ``x``.
    """
    if not enabled:
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
    sigma_relative: float,
    *,
    unit: float,
    floor: float | None = None,
    enabled: bool,
) -> Tensor:
    """Add Pelgrom-scaled Gaussian mismatch to a binary-weighted ladder.

    Per-cell matching sigma: ``σ_k = √(X_k / X_unit) · σ_u · X_unit``
    where ``σ_u`` is the per-unit-cell relative sigma.

    Args:
        ideal: Tensor of nominal per-cell values.
        sigma_relative: Per-unit-cell relative sigma ``σ_u``.
        unit: Single-unit-cell value ``X_unit`` in the same units as
            ``ideal``.
        floor: Optional minimum clamp applied after sampling.
        enabled: Master toggle. ``False`` returns ``ideal`` unchanged.

    Returns:
        Tensor with the same shape / dtype / device as ``ideal``.
    """
    if not enabled:
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
    does not already inject per-cycle randomness. Output is clamped to
    ``[0, 2 ** n_bits - 1]``.

    Args:
        code: Integer code tensor.
        n_bits: Active bit width (used for the upper clamp).
        enabled: Master toggle. ``False`` returns ``code`` unchanged.

    Returns:
        Jittered code with the same dtype / device as ``code``.
    """
    if not enabled:
        return code
    max_code = (1 << n_bits) - 1
    jitter = torch.randint(low=0, high=2, size=code.shape, dtype=code.dtype, device=code.device)
    return (code + jitter).clamp(min=0, max=max_code)
