"""Abstract base class for TIA models.

See also:
    docs/reference/primitive/analog/tia/README.md
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import Tensor

from neurox.common.mixin import RegistryMixin
from neurox.primitive.analog.base import AnalogBase, AnalogConfig, AnalogPolicy


@dataclass(frozen=True)
class TIAConfig(AnalogConfig):
    """Base configuration for TIA implementations.

    Attributes:
        area_per_inst__um2: Silicon area per fabricated instance.
        leakage_per_inst__uW: Static leakage per instance.
    """

    area_per_inst__um2: float
    leakage_per_inst__uW: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_ppa()

    def validate_ppa(self) -> None:
        self._require_non_neg(self.area_per_inst__um2, "area_per_inst__um2")
        self._require_non_neg(self.leakage_per_inst__uW, "leakage_per_inst__uW")


@dataclass(frozen=True)
class TIAPolicy(AnalogPolicy):
    """Abstract marker base for TIA-family nonideality policies."""


@dataclass(frozen=True)
class TIASnap:
    """Marker base for per-call snaps of a TIA's fabricated state."""


SnapT = TypeVar("SnapT", bound=TIASnap)


class TIA(
    AnalogBase[TIAConfig, TIAPolicy],
    RegistryMixin[type["TIAConfig"], "TIA"],
    Generic[SnapT],
):
    """Abstract base for transimpedance-amp clamp drivers."""

    def __init__(
        self,
        *,
        config: TIAConfig,
        policy: TIAPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        """Register the instance with :class:`nn.Module` and the profiler."""
        del dtype, T__K  # captured by the subclass init
        super().__init__(config=config, policy=policy, name=name, inst_shape=inst_shape)

    @classmethod
    def from_config(
        cls,
        *,
        config: TIAConfig,
        policy: TIAPolicy,
        name: str,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> TIA:
        """Build the concrete impl registered for ``type(config)``."""
        impl = cls._lookup_impl(type(config))
        return impl(
            config=config,
            policy=policy,
            name=name,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )

    @abstractmethod
    def snapshot(
        self,
        *,
        v_ref__V: Tensor,
        shape: tuple[int, ...],
        multi_coords: tuple[Tensor, ...] | None,
    ) -> SnapT:
        """Sample one per-call runtime snap over ``shape``.

        Args:
            v_ref__V: Injected reference clamp voltage. A scalar or
                instance-shaped tensor that broadcasts onto ``shape``;
                stored in the returned snap.
            shape: Per-call broadcast shape; the snap fills tensor
                fields at this shape.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view; ``None`` returns the
                full view.

        Returns:
            Per-call snap of the fabricated state.
        """
        raise NotImplementedError

    @abstractmethod
    def solve_clamp(
        self,
        i_port__uA: Tensor,
        snap: SnapT,
        *,
        v_clamp_init__V: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Boundary clamp solve: returns ``(v_clamp__V, dVclamp_dI__MOhm)``."""
        raise NotImplementedError
