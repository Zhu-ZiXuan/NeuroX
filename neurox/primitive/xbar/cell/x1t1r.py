"""Abstract 1T1R crossbar cell — shared types and electrical interface.

See Also:
    docs/reference/primitive/xbar/cell/1t1r.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
from torch import Tensor

from neurox.common import ConfigBase, DcopBase, ModuleBase, PolicyBase, RegistryMixin, SnapBase


class XbarCell1t1rConfig(ConfigBase, ABC):
    pass


class XbarCell1t1rPolicy(PolicyBase, ABC):
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


class XbarCell1t1r[ConfigT: XbarCell1t1rConfig, PolicyT: XbarCell1t1rPolicy, SnapT: XbarCell1t1rSnap](
    ModuleBase[ConfigT, PolicyT],
    RegistryMixin[
        XbarCell1t1rConfig,
        XbarCell1t1rPolicy,
        "XbarCell1t1r[XbarCell1t1rConfig, XbarCell1t1rPolicy, Any]",
    ],
    ABC,
):
    is_profile_target: ClassVar[bool] = False

    def __init__(
        self,
        *,
        config: ConfigT,
        policy: PolicyT,
        inst_shape: tuple[int, ...],
        dtype: torch.dtype,
        T__K: float,
    ) -> None:
        del dtype, T__K
        super().__init__(config=config, policy=policy, inst_shape=inst_shape)

    @property
    @abstractmethod
    def w_state_num(self) -> int:
        """Number of programmable weight states."""
        raise NotImplementedError

    @abstractmethod
    def snapshot(
        self,
        *,
        control: Tensor,
        shape: tuple[int, ...],
        t_elapsed: float,
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
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        snap: SnapT,
    ) -> XbarCell1t1rDcop:
        """Return the branch operating point including the access node."""
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
