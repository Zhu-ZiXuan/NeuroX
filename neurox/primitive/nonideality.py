"""Reusable analog non-ideality kernels and their config dataclasses.

Every kernel owns its own enable branch through a keyword-only `enabled` flag, a plain
Python `bool` that resolves at trace time, so a disabled source is expressed by the flag
alone and never by a `None` argument. A disabled kernel returns the input object itself
rather than a clone — a wide broadcast view stays unmaterialized — so a caller treats
every result as read-only. A kernel whose spread is a single scalar takes that scalar
directly, while one with several coupled parameters takes a frozen config declared beside
it, keeping that parameter set named and validated once.

See Also:
    docs/reference/primitive/nonideality.md
    docs/system_design/nonideality_kernels.md
"""

import torch
from torch import Tensor

from neurox.common import ConfigBase


class StuckAtFaultConfig(ConfigBase):
    """Stuck-at fault probabilities."""

    p_at_min: float
    p_at_max: float

    def validate(self) -> None:
        self._require_non_neg(self.p_at_min, "p_at_min")
        self._require_non_neg(self.p_at_max, "p_at_max")
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
    """Replace cells with stuck-at-min / stuck-at-max values."""
    if not enabled:
        return x
    rand_mask = torch.rand_like(x)
    p_min = config.p_at_min
    p_both = config.p_at_min + config.p_at_max
    is_min = rand_mask < p_min
    is_max = (rand_mask >= p_min) & (rand_mask < p_both)
    return torch.where(is_min, min_val, torch.where(is_max, max_val, x))


def apply_gaussian(x: Tensor, sigma: float | Tensor, *, enabled: bool) -> Tensor:
    """Apply additive Gaussian noise."""
    if not enabled:
        return x
    return x + torch.randn_like(x) * sigma


def apply_relative_gaussian(x: Tensor, sigma_relative: float, *, enabled: bool) -> Tensor:
    """Apply multiplicative Gaussian noise proportional to the signal.

    The multiplicative form keeps an exact zero exact.
    """
    if not enabled:
        return x
    return x * (1.0 + torch.randn_like(x) * sigma_relative)


class StateDependentGaussianConfig(ConfigBase):
    """State-dependent Gaussian noise config."""

    sigma_slope: float
    """Linear growth of the noise σ per unit of `|x|`."""
    sigma_intercept: float
    """Base σ at `|x| = 0`."""

    def validate(self) -> None:
        self._require_non_neg(self.sigma_slope, "sigma_slope")
        self._require_non_neg(self.sigma_intercept, "sigma_intercept")


def apply_state_dependent_gaussian(
    x: Tensor,
    config: StateDependentGaussianConfig,
    *,
    enabled: bool,
) -> Tensor:
    """Apply Gaussian noise whose σ scales with the magnitude of `x`."""
    if not enabled:
        return x
    sigma = config.sigma_slope * x.abs() + config.sigma_intercept
    return x + torch.randn_like(x) * sigma


class LognormalConfig(ConfigBase):
    """Multiplicative log-normal noise config."""

    sigma: float
    """Standard deviation of the underlying normal, not of the multiplicative factor."""

    def validate(self) -> None:
        self._require_non_neg(self.sigma, "sigma")


def apply_lognormal(x: Tensor, config: LognormalConfig, *, enabled: bool) -> Tensor:
    """Apply multiplicative log-normal noise."""
    if not enabled:
        return x
    return x * torch.exp(torch.randn_like(x) * config.sigma)


class StateDependentLognormalConfig(ConfigBase):
    """State-dependent log-normal noise config."""

    sigma_slope: float
    """Amount the noise σ falls as the normalised state rises from 0 to 1."""
    sigma_intercept: float
    """Base σ at the min state (normalised state 0)."""
    min_val: float
    """Lower bound of the state-normalisation range."""
    max_val: float
    """Upper bound of the state-normalisation range."""

    def validate(self) -> None:
        self._require_non_neg(self.sigma_slope, "sigma_slope")
        self._require_non_neg(self.sigma_intercept, "sigma_intercept")
        if not (self.max_val > self.min_val):
            raise ValueError(f"require: max_val ({self.max_val}) > min_val ({self.min_val})")


def apply_state_dependent_lognormal(
    x: Tensor,
    config: StateDependentLognormalConfig,
    *,
    enabled: bool,
) -> Tensor:
    """Apply log-normal noise whose σ depends on normalised state."""
    if not enabled:
        return x
    x_norm = (x - config.min_val) / (config.max_val - config.min_val + 1e-12)
    sigma = x_norm * (-config.sigma_slope) + config.sigma_intercept
    return x * torch.exp(torch.randn_like(x) * sigma)


class GammaConfig(ConfigBase):
    """Multiplicative Gamma noise config (constant shape and scale)."""

    shape_k: float
    scale_theta: float

    def validate(self) -> None:
        self._require_pos(self.shape_k, "shape_k")
        self._require_pos(self.scale_theta, "scale_theta")


def apply_gamma_noise(x: Tensor, config: GammaConfig, *, enabled: bool) -> Tensor:
    """Apply multiplicative Gamma noise normalised to unit mean."""
    if not enabled:
        return x
    gamma_dist = torch.distributions.Gamma(config.shape_k, 1.0 / config.scale_theta)
    gamma_sample = gamma_dist.sample(x.shape).to(x.device, x.dtype)
    mean = config.shape_k * config.scale_theta
    return x * (gamma_sample / mean)


class StateDependentGammaConfig(ConfigBase):
    """State-dependent Gamma noise config."""

    k_slope: float
    """Rate at which the Gamma shape k varies with normalised state."""
    k_intercept: float
    """Gamma shape k at the min state (normalised state 0)."""
    theta: float
    """Scale parameter, held constant across all states."""
    min_val: float
    """Lower bound of the state-normalisation range."""
    max_val: float
    """Upper bound of the state-normalisation range."""

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

    The Gamma sampler requires float32 or higher; a lower-precision input is
    cast for sampling and cast back on return.
    """
    if not enabled:
        return x
    in_dtype = x.dtype
    needs_cast = in_dtype not in (torch.float32, torch.float64)
    x32 = x.float() if needs_cast else x

    # --- 1: normalize the conductance state ---

    x_norm = (x32 - config.min_val) / (config.max_val - config.min_val + 1e-12)

    # --- 2: derive the state-dependent shape ---

    k = (x_norm * config.k_slope + config.k_intercept).clamp(min=0.1)

    # --- 3: sample elementwise gamma noise ---

    theta_tensor = torch.full(x32.shape, config.theta, dtype=x32.dtype, device=x32.device)
    rate = 1.0 / theta_tensor
    gamma_sample = torch.distributions.Gamma(concentration=k, rate=rate).sample()

    # --- 4: normalize to unit-mean gain ---

    mean = (k * theta_tensor).clamp(min=1e-12)
    result = x32 * (gamma_sample / mean)
    return result.to(in_dtype) if needs_cast else result


class TelegraphConfig(ConfigBase):
    """Random telegraph noise config."""

    amplitude_mean: float
    amplitude_std: float
    """Standard deviation of the Gaussian amplitude draw."""
    p_high_state: float
    """Probability that a cell sits in the high RTN state."""

    def validate(self) -> None:
        self._require_non_neg(self.amplitude_std, "amplitude_std")
        if not (0.0 <= self.p_high_state <= 1.0):
            raise ValueError(f"require: 0 <= p_high_state ({self.p_high_state}) <= 1")


def apply_telegraph_noise(x: Tensor, config: TelegraphConfig, *, enabled: bool) -> Tensor:
    """Apply random telegraph noise."""
    if not enabled:
        return x
    amplitude = torch.randn_like(x) * config.amplitude_std + config.amplitude_mean
    sign = torch.where(torch.rand_like(x) < 0.5, -1.0, 1.0).to(dtype=x.dtype)
    mask = (torch.rand_like(x) < config.p_high_state).to(dtype=x.dtype)
    return x + amplitude * sign * mask


def apply_pelgrom_mismatch(
    ideal: Tensor,
    sigma_relative: float,
    *,
    unit: float,
    floor: float | None = None,
    enabled: bool,
) -> Tensor:
    """Add Pelgrom-scaled Gaussian mismatch to a binary-weighted ladder.

    Each cell's absolute matching σ is the area-scaled Pelgrom spread.

    Args:
        ideal: Tensor of nominal per-cell values.
        sigma_relative: Per-unit-cell relative σ.
        unit: Single-unit-cell value in the same units as `ideal`.
        floor: Optional minimum clamp applied after sampling.
        enabled: Master toggle; `False` returns `ideal` unchanged.

    Returns:
        Tensor with the same dtype / device as `ideal`.
        Shape: `[...]`.
    """
    if not enabled:
        return ideal
    sigma = torch.sqrt(ideal / unit) * (sigma_relative * unit)
    out = ideal + torch.randn_like(ideal) * sigma
    if floor is not None:
        out = out.clamp_min(floor)
    return out


def apply_lsb_jitter(
    code: Tensor,
    *,
    unsigned_max: int,
    enabled: bool,
) -> Tensor:
    """Add a Bernoulli(0.5) 0/+1 LSB jitter to an integer code.

    Output is clamped to `[0, unsigned_max]`.
    """
    if not enabled:
        return code
    jitter = torch.randint(low=0, high=2, size=code.shape, dtype=code.dtype, device=code.device)
    return (code + jitter).clamp(min=0, max=unsigned_max)
