"""Abstract 1T1R crossbar cell — shared types and electrical interface.

See Also:
    docs/reference/primitive/xbar/cell/1t1r.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor

from neurox.common.module import ConfigBase, DcopBase, NonProfileModule, PolicyBase, SnapBase
from neurox.common.registry_mixin import RegistryMixin


class XbarCell1t1rConfig(ConfigBase, base_only=True):
    pass


class XbarCell1t1rPolicy(PolicyBase, base_only=True):
    pass


class XbarCell1t1rSnap(SnapBase):
    v_wl__V: Tensor
    """Word-line drive voltage at each cell's own NMOS gate."""


class XbarCell1t1rDcop(DcopBase):
    i__uA: Tensor
    """Branch current, positive bit-line into source-line."""
    di_dvbl__uS: Tensor
    """BL-side branch conductance ∂I/∂V_BL, non-negative."""
    di_dvsl__uS: Tensor
    """SL-side branch conductance ∂I/∂V_SL, non-positive."""
    v_x__V: Tensor
    """Access-node voltage at the NMOS drain / RRAM bottom."""


_Config = XbarCell1t1rConfig
_Policy = XbarCell1t1rPolicy
_Snap = XbarCell1t1rSnap
_Dcop = XbarCell1t1rDcop


class XbarCell1t1r[SnapT: _Snap](NonProfileModule, RegistryMixin[_Config, _Policy], ABC, base_only=True):
    config: _Config
    policy: _Policy

    def __init__(
        self,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    # === Public API ===

    @classmethod
    def from_config(
        cls,
        *,
        config: _Config,
        policy: _Policy,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> XbarCell1t1r[Any]:
        """Build the 1T1R cell registered for the config-policy pair.

        Returns:
            Registered 1T1R cell implementation.
        """
        impl = cls._lookup_impl(config=config, policy=policy)
        return impl(
            config=config,
            policy=policy,
            inst_shape=inst_shape,
            dtype=dtype,
        )

    # === For subclass to implement or override ===

    @property
    @abstractmethod
    def w_state_num(self) -> int:
        """Number of programmable weight states."""
        raise NotImplementedError

    @abstractmethod
    def snapshot(
        self,
        control: Tensor,
        *,
        shape: tuple[int, ...],
    ) -> SnapT:
        """Sample the cell state and word-line control for one solve."""
        raise NotImplementedError

    @abstractmethod
    def program(self, w_state_idx: Tensor) -> None:
        """Program the storage device from a state-index tensor."""
        raise NotImplementedError

    @abstractmethod
    def solve_dc(
        self,
        *,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        snap: SnapT,
    ) -> _Dcop:
        """Return the branch operating point including the access node."""
        raise NotImplementedError
