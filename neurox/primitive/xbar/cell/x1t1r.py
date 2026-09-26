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
    """Implement a programmable 1T1R branch for repeated DC evaluation.

    Concrete classes supply `w_state_num`, `program`, `snapshot`, and
    `solve_dc`. Register a config-policy pair on this family and initialize
    `NonProfileModule` through this constructor. The cell owns electrical state;
    its area and leakage are accounted for by a containing circuit rather than
    reported independently.

    `program` maps logical state indices onto per-instance storage. `snapshot`
    combines that state with word-line control and any access-level randomness.
    `solve_dc` reuses the supplied snapshot across trial BL/SL voltages and
    returns branch current, both terminal derivatives, and the internal
    access-node voltage. Keep these evaluations deterministic for a fixed
    snapshot; do not resample or reprogram during a numerical solve.

    Args:
        config: Hardware configuration.
        policy: Run policy matching `config`.
        inst_shape: Positive physical instance extents; singletons allow
            broadcasting.
        dtype: Electrical tensor dtype.
    """

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
        """Capture programmed state and word-line drive for one physical access.

        Sample access variation once at the requested layout. Returned tensor
        fields must broadcast with the BL/SL voltage layout accepted by
        `solve_dc`; expanded views should be treated as read-only. Preserve that
        snapshot for every numerical iteration of the same access.

        Args:
            control: Word-line drive voltage, broadcastable to the evaluation
                layout.
            shape: Full cell evaluation layout, including independent accesses
                and the cell instance axes. Existing programmed state must
                expand to it.

        Returns:
            The implementation's snapshot carrying control and held branch
            parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def program(self, w_state_idx: Tensor) -> None:
        """Replace per-cell storage using logical state indices.

        Map indices in `[0, w_state_num - 1]` onto this model's physical storage
        and apply programming variation at this event. Validate the instance
        layout and prepare any derived parameters used by snapshots. Call after
        device placement and required fabrication, outside compiled electrical
        evaluation.

        Args:
            w_state_idx: Integer state selection for every physical cell
                instance.
                Shape: `[*inst_shape]`.
        """
        raise NotImplementedError

    @abstractmethod
    def solve_dc(
        self,
        *,
        v_bl__V: Tensor,
        v_sl__V: Tensor,
        snap: SnapT,
    ) -> _Dcop:
        """Evaluate a held branch realization at the supplied terminal voltages.

        Inputs and snapshot fields must broadcast to the same cell evaluation
        layout. Return BL-to-SL current with its local derivatives and
        access-node voltage. The BL derivative is nonnegative and the SL
        derivative nonpositive under the family's passive-branch model. Evaluate
        derivatives at this operating point; do not replace them with a nominal
        conductance unless the model is linear. The solver may call this
        repeatedly with the same snapshot and new voltages.

        Args:
            v_bl__V: Bit-line terminal voltages for the evaluated cells.
            v_sl__V: Source-line terminal voltages under the same broadcast
                layout.
            snap: Held cell realization sampled once for this physical access.

        Returns:
            Branch operating point containing current, terminal derivatives, and
            access-node voltage.
        """
        raise NotImplementedError
