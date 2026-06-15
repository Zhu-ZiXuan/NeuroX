"""Abstract base class for TIA models.

See also:
    docs/dev/modules/analog/tia/README.md
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import Tensor

from neurox.common.circuit import CircuitBase, CircuitConfig
from neurox.common.mixin import RegistryMixin


@dataclass(frozen=True)
class TIAConfig(CircuitConfig):
    """Base configuration for TIA implementations.

    Attributes:
        v_ref__V: Reference clamp voltage [V].
    """

    v_ref__V: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.validate_ppa()


@dataclass(frozen=True)
class TIAPolicy:
    """Abstract marker base for TIA-family nonideality policies."""


@dataclass(frozen=True)
class TIASnapshot:
    """Marker base for per-call snapshots of a TIA's fabricated state."""


SnapshotT = TypeVar("SnapshotT", bound=TIASnapshot)


class TIA(
    CircuitBase[TIAConfig],
    RegistryMixin[type["TIAConfig"], "TIA"],
    Generic[SnapshotT],
):
    """Abstract base for transimpedance-amp clamp drivers.

    Parameterised by the concrete snapshot type ``SnapshotT`` so each
    implementation declares its snapshot dataclass exactly once and
    ``snapshot`` / ``solve_clamp`` carry that concrete type without an
    LSP-narrowing override. The registry-impl slot is unparameterised
    because Python generics are invariant — each concrete impl binds
    ``SnapshotT`` to its own snapshot subclass.
    """

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
        del policy, dtype, T__K  # captured by the subclass init
        super().__init__(config=config, name=name, inst_shape=inst_shape)

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

    @property
    @abstractmethod
    def v_ref__V(self) -> float:
        """Reference voltage [V]."""
        raise NotImplementedError

    @abstractmethod
    def snapshot(self, *, shape: tuple[int, ...], multi_coords: tuple[Tensor, ...] | None) -> SnapshotT:
        """Sample one per-call runtime snapshot over ``shape``.

        Args:
            shape: Per-call broadcast shape; the snapshot fills tensor
                fields at this shape.
            multi_coords: Advanced-index tuple selecting a chunk's
                positions from the broadcast view; ``None`` returns the
                full view.

        Returns:
            Per-call snapshot of the fabricated state.
        """
        raise NotImplementedError

    @abstractmethod
    def solve_clamp(
        self,
        i_port__uA: Tensor,
        snapshot: SnapshotT,
        *,
        v_clamp_init__V: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Boundary clamp solve: returns ``(v_clamp__V, dVclamp_dI__MOhm)``."""
        raise NotImplementedError
