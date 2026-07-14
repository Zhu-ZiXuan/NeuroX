"""Ideal current-mirror — single-ended ratio-copy transport primitive.

See also:
    docs/reference/primitive/analog/current_mirror.md
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


@dataclass(frozen=True, kw_only=True)
class CurrentMirrorConfig(AnalogConfig):
    """Immutable configuration for :class:`CurrentMirror`.

    Attributes:
        mirror_ratio: Dimensionless output/input copy ratio.
        ratio_sigma_relative: Relative (Pelgrom) σ of the static
            per-instance mirror-ratio mismatch [dimensionless]; ``0``
            leaves the exact ratio copy.
    """

    # --- Copy ratio ---
    mirror_ratio: float

    # --- Mismatch ---
    ratio_sigma_relative: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_ratio()

    def validate_ratio(self) -> None:
        self._require_pos(self.mirror_ratio, "mirror_ratio")
        self._require_non_neg(self.ratio_sigma_relative, "ratio_sigma_relative")


@dataclass(frozen=True)
class CurrentMirrorPolicy(AnalogPolicy):
    """Per-source toggles selecting which CurrentMirror nonidealities are active.

    Attributes:
        mismatch: Apply the static per-instance (Pelgrom) copy-ratio mismatch
            with σ ``ratio_sigma_relative``; ``False`` leaves the exact ratio
            copy.
    """

    mismatch: bool


class CurrentMirror(AnalogBase[CurrentMirrorConfig, CurrentMirrorPolicy]):
    """Single-ended current mirror — pure ratio-copy transport primitive.

    The copy is exact at ``mirror_ratio`` unless the ``mismatch`` policy is on,
    in which case the ratio carries a static per-instance multiplicative
    (Pelgrom) Gaussian with relative σ ``ratio_sigma_relative``.

    Args:
        config: Concrete configuration dataclass.
        policy: Per-source nonideality enable flags.
        name: Hierarchical instance name used by the profiler.
        inst_shape: Per-instance fabrication shape.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    # Non-reporter: embedded primitive whose static PPA rolls up to the owning block.
    reports_static_ppa: ClassVar[bool] = False

    ratio_mismatch: Tensor

    def __init__(
        self,
        *,
        config: CurrentMirrorConfig,
        policy: CurrentMirrorPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, name=name, inst_shape=inst_shape)
        self.dtype = dtype
        self.T__K = T__K

        # Held static per-instance copy-ratio mismatch multiplier (Pelgrom);
        # unit ratio when the mismatch is off.
        self.register_buffer(
            "ratio_mismatch",
            torch.ones(inst_shape, dtype=dtype),
            persistent=False,
        )

    def _sample_fabricate_mismatch(self) -> None:
        """Sample the static per-instance copy-ratio mismatch over ``inst_shape``.

        A multiplicative (Pelgrom) Gaussian ``1 + N(0, ratio_sigma_relative)``
        per mirror element; ``mismatch`` off leaves the exact unit ratio.
        """
        if self.policy.mismatch:
            self.ratio_mismatch = 1.0 + torch.randn_like(self.ratio_mismatch) * self.config.ratio_sigma_relative
        else:
            self.ratio_mismatch = torch.ones_like(self.ratio_mismatch)

    def replicate(self, i_in__uA: Tensor) -> Tensor:
        """Copy the input current at the configured mirror ratio.

        Scales the nominal ``mirror_ratio`` by the static per-instance
        copy-ratio mismatch (unit when ``mismatch`` is off).

        Args:
            i_in__uA: Input branch current, shape ``(*leading, *inst_shape)``.

        Returns:
            Output branch current ``mirror_ratio * ratio_mismatch * i_in__uA``.
        """
        return self.config.mirror_ratio * self.ratio_mismatch * i_in__uA
