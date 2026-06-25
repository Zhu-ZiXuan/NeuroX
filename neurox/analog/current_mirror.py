"""Ideal current-mirror — single-ended ratio-copy behavioural block.

See also:
    docs/reference/analog/current_mirror.md
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from neurox.common.circuit import CircuitBase, CircuitConfig


@dataclass(frozen=True, kw_only=True)
class CurrentMirrorConfig(CircuitConfig):
    """Immutable configuration for :class:`CurrentMirror`.

    Attributes:
        mirror_ratio: Dimensionless output/input copy ratio.
        v_supply__V: Rail supply voltage [V] driving the data-dependent
            output-branch dissipation.
        ratio_sigma_relative: Relative (Pelgrom) mirror-ratio mismatch sigma
            [dimensionless], gated by the ``mismatch`` policy; ``0`` leaves the
            exact ratio copy.
        area_per_inst__um2: Silicon area per fabricated instance [um²].
        leakage_per_inst__uW: Static leakage per instance [uW].
    """

    # --- Copy ratio ---
    mirror_ratio: float

    # --- Rail ---
    v_supply__V: float

    # --- Mismatch ---
    ratio_sigma_relative: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_ratio()
        self.validate_rail()
        self.validate_ppa()

    def validate_ratio(self) -> None:
        self._require_pos(self.mirror_ratio, "mirror_ratio")
        self._require_nonneg(self.ratio_sigma_relative, "ratio_sigma_relative")

    def validate_rail(self) -> None:
        self._require_pos(self.v_supply__V, "v_supply__V")


@dataclass(frozen=True)
class CurrentMirrorPolicy:
    """Per-source toggles selecting which CurrentMirror nonidealities are active.

    Attributes:
        mismatch: Apply a multiplicative (Pelgrom) Gaussian on the copy ratio
            with sigma ``ratio_sigma_relative``; ``False`` leaves the exact
            ratio copy.
    """

    mismatch: bool


class CurrentMirror(CircuitBase[CurrentMirrorConfig]):
    """Single-ended current mirror — ratio copy with data-dependent rail energy.

    The copy is exact at ``mirror_ratio`` unless the ``mismatch`` policy is on,
    in which case the ratio is perturbed by a per-call multiplicative (Pelgrom)
    Gaussian ``(1 + N(0, ratio_sigma_relative))``. The rail dissipation counts
    the output branch only.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        name: Hierarchical instance name used by the profiler.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature [K].
        read_pulse__ns: Read-window width [ns] passed by the caller;
            scales the per-call rail energy.
    """

    def __init__(
        self,
        *,
        config: CurrentMirrorConfig,
        policy: CurrentMirrorPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
        read_pulse__ns: float,
    ) -> None:
        super().__init__(config=config, name=name, inst_shape=inst_shape)
        self.policy = policy
        self.dtype = dtype
        self.T__K = T__K
        self.read_pulse__ns = read_pulse__ns

    def replicate(self, i_in__uA: Tensor) -> Tensor:
        """Copy the input current at the configured mirror ratio.

        When ``mismatch`` is enabled, the ratio is perturbed per call by a
        multiplicative (Pelgrom) Gaussian ``mirror_ratio · (1 + N(0,
        ratio_sigma_relative))`` sampled element-wise (per-call ``randn_like``,
        not a fixed per-instance buffer); otherwise the scalar ratio is used.

        Args:
            i_in__uA: Input branch current [uA].

        Returns:
            Output branch current ``ratio * i_in__uA`` [uA].
        """
        ratio: Tensor | float
        if self.policy.mismatch:
            ratio = self.config.mirror_ratio * (1.0 + torch.randn_like(i_in__uA) * self.config.ratio_sigma_relative)
        else:
            ratio = self.config.mirror_ratio
        i_out__uA = ratio * i_in__uA

        # Rail dissipation on the OUTPUT branch only: V_supply·|i_out|·t. The
        # input current is sourced externally (its production energy is
        # accounted by the upstream block), so it is NOT counted here. The
        # energy naturally tracks the (possibly perturbed) i_out. uA·V·ns = fJ.
        dynamic_energy__fJ = self.config.v_supply__V * i_out__uA.abs() * self.read_pulse__ns
        self._log_dynamic_energy(dynamic_energy__fJ)
        return i_out__uA
