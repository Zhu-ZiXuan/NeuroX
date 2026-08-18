"""Abstract 1T1R crossbar cell — shared config and result types.

See Also:
    docs/reference/primitive/xbar/cell/1t1r.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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
    pass


class XbarCell1t1rPolicy(XbarCellPolicy, ABC):
    pass


class XbarCell1t1rSnap(XbarCellSnap):
    v_wl__V: Tensor
    """Word-line drive voltage at each cell's own NMOS gate. Shape: `[..., col, row]`."""


class XbarCell1t1rDcop(XbarCellDcop):
    v_x__V: Tensor
    """Access-node voltage at the NMOS drain / RRAM bottom. Shape: `[..., col, row]`."""


class XbarCell1t1r[ConfigT: XbarCell1t1rConfig, PolicyT: XbarCell1t1rPolicy, SnapT: XbarCell1t1rSnap](
    XbarCell[ConfigT, PolicyT, SnapT, XbarCell1t1rDcop],
    RegistryMixin[
        XbarCell1t1rConfig,
        XbarCell1t1rPolicy,
        "XbarCell1t1r[XbarCell1t1rConfig, XbarCell1t1rPolicy, Any]",
    ],
    ABC,
):
    """Base class for condensed series access-device and storage cells."""

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
