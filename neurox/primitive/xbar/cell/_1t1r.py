"""Abstract 1T1R crossbar cell — shared config and result types.

See also:
    docs/reference/primitive/xbar/cell/_1t1r/cell.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch
from torch import Tensor

from neurox.common import RegistryMixin

from .base import (
    XbarCell,
    XbarCellConfig,
    XbarCellDcop,
    XbarCellPolicy,
    XbarCellSnap,
)


class XbarCell1t1rConfig(XbarCellConfig, ABC):
    """Config base of the 1T1R cell family.

    The family shares no config field: a 1T1R model states its own branch
    knobs and nothing else. The class is the registry and dispatch anchor
    every concrete 1T1R configuration derives from.
    """


class XbarCell1t1rPolicy(XbarCellPolicy, ABC):
    """Base policy for 1T1R cell nonidealities."""


@dataclass(frozen=True, kw_only=True)
class XbarCell1t1rSnap(XbarCellSnap):
    """Per-call snap base of a 1T1R cell's fabricated state.

    Attributes:
        v_wl__V: Word-line drive voltage at each cell's own NMOS gate. The
            field states the cell's control terminal, whatever wiring put
            the voltage there, so the producing array lays it out on the
            cell grid rather than asserting one value per row.
            Shape: ``[..., col, row]``.
    """

    v_wl__V: Tensor


@dataclass(frozen=True)
class XbarCell1t1rDcop(XbarCellDcop):
    """1T1R branch working point with the condensed access-node voltage.

    Attributes:
        v_x__V: Access-node voltage (NMOS drain / RRAM bottom).
            Shape: ``[..., col, row]``.
    """

    v_x__V: Tensor


CellSnapT = TypeVar("CellSnapT", bound=XbarCell1t1rSnap)
ConfigT = TypeVar("ConfigT", bound=XbarCell1t1rConfig, covariant=True)
PolicyT = TypeVar("PolicyT", bound=XbarCell1t1rPolicy, covariant=True)


class XbarCell1t1r(
    XbarCell[ConfigT, PolicyT, CellSnapT, XbarCell1t1rDcop],
    RegistryMixin[
        XbarCell1t1rConfig,
        XbarCell1t1rPolicy,
        "XbarCell1t1r[XbarCell1t1rConfig, XbarCell1t1rPolicy, Any]",
    ],
    Generic[ConfigT, PolicyT, CellSnapT],
    ABC,
):
    """Base class for condensed series access-device and storage cells.

    Args:
        config: Concrete 1T1R cell configuration.
        policy: Composite per-device nonideality policy.
        inst_shape: Per-instance shape ``(..., col, row)``.
        dtype: Tensor dtype for internal buffers.
        T__K: Operating temperature.
    """

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape, dtype=dtype, T__K=T__K)

    @property
    @abstractmethod
    def w_state_num(self) -> int:
        """Number of programmable weight states."""
        raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        *,
        config: XbarCell1t1rConfig,
        policy: XbarCell1t1rPolicy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> XbarCell1t1r[XbarCell1t1rConfig, XbarCell1t1rPolicy, Any]:
        """Build the 1T1R cell registered for the config-policy pair.

        Args:
            config: Concrete 1T1R cell configuration.
            policy: Composite per-device nonideality policy.
            inst_shape: Per-instance shape ``(..., col, row)``.
            dtype: Tensor dtype for internal buffers.
            T__K: Operating temperature.

        Returns:
            Registered 1T1R cell implementation.
        """
        impl = cls._lookup_neurox_module(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
            T__K=T__K,
        )
